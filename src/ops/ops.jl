using FFTW, DSP, ImageFiltering, NNlib, Flux, CUDA
using Images, ColorVectorSpace, ColorTypes

include("../utilities/base_funcs.jl")


"""
    tvd_fft(y::Array{<:Real,N}, λ, ρ=1; maxit=100, tol=1e-2, verbose=true)

2D anisotropic TV denoising with periodic boundary conditions via ADMM. 
Accepts 2D, 3D, 4D tensors in (H,W,C,B) form, where the last two
dimensions are optional.
"""
HT(x,τ) = x*(abs(x) > τ);                      # hard-thresholding
ST(x,τ) = sign.(x).*max.(abs.(x).-τ, 0);       # soft-thresholding
pixelnorm(x) = sqrt.(sum(abs2, x, dims=(3,4))) # 2-norm on 4D image-tensor pixel-vectors
BT(x,τ) = max.(1 .- τ ./ pixelnorm(x), 0).*x   # block-thresholding
objfun_iso(x,Dx,y,λ)   = 0.5*sum(abs2.(x-y)) + λ*norm(pixelnorm(Dx), 1) 
objfun_aniso(x,Dx,y,λ) = 0.5*sum(abs2.(x-y)) + λ*norm(Dx, 1)

function fdkernel(T::Type)
	dx_prime = cat(cat(ones(T, 1), zeros(T, 1), dims=1), cat(-ones(T, 1), zeros(T, 1), dims=1), dims=2)
	dx = cat(dx_prime, cat(cat(ones(T, 1), -ones(T, 1), dims=1), cat(zeros(T, 1), zeros(T, 1), dims=1), dims=2), dims=4)
	dy = reverse(permutedims(dx, (2,1,4,3)), dims=:);
	return dx, dy
end


function tvd_fft(y::AbstractArray{T}, λ, ρ=1; h::AbstractArray{T}=[], isotropic=false, maxit=100) where {T}
	M, N, P, _ = size(y)
	y = permutedims(y, (1,2,4,3)) # move channels to batch dimension
	τ = λ / ρ;

	if isempty(h)
		Σ = 1
	else
		hh = NNlib.pad_constant(h, (0, M-size(h)[1], 0, N-size(h)[2], 0, 0, 0, 0))
		# Σ = CUDA.CUFFT.rfft(hh)
		Σ_ref = CUDA.CUFFT.rfft(hh[:,:,1,1])
		Σ_ref = cat(Σ_ref, zeros(eltype(Σ_ref), size(Σ_ref)), dims=5)
		Σ = Σ_ref[:,:,:,:,1]
	end

	# precompute C for x-update
	Λx = CUDA.CUFFT.rfft([1 -1 zeros(N-2)'; zeros(M-1,N)]);
	Λy = CUDA.CUFFT.rfft([[1; -1; zeros(M-2)] zeros(M,N-1)])
	C = 1 ./ ( abs2.(Σ) .+ ρ.*(abs2.(Λx) .+ abs2.(Λy)) )

	if isotropic
		objfun = (x,Dx) -> objfun_iso(x,Dx,y,λ)
		thresh_type = BT # block-thresholding
	else
		objfun = (x,Dx) -> objfun_aniso(x,Dx,y,λ)
		thresh_type = ST # soft-thresholding
	end

	# initialization
	x::AbstractArray{T} = zeros(T, M,N,1,P)
	Dxᵏ::AbstractArray{T} = zeros(T, M,N,2,P)
	z::AbstractArray{T} = zeros(T, M,N,2,P)
	u::AbstractArray{T} = zeros(T, M,N,2,P)

	# conv kernel
	W, Wᵀ = fdkernel(T)

	# (in-place) Circular convolution
	cdims = DenseConvDims(NNlib.pad_circular(x, (1,0,1,0)), W)
	cdimsᵀ= DenseConvDims(NNlib.pad_circular(z, (0,1,0,1)), Wᵀ)
	D(x) = NNlib.conv(NNlib.pad_circular(x, (1,0,1,0)), W, cdims)
	Dᵀ(z) = NNlib.conv(NNlib.pad_circular(z, (0,1,0,1)), Wᵀ,cdimsᵀ)

	if isempty(h)
		H = identity
		Hᵀ= identity
	else
		hᵀ= reverse(h)
		padu, padd = ceil(Int,(size(h,1)-1)/2), floor(Int,(size(h,1)-1)/2)
		padl, padr = ceil(Int,(size(h,2)-1)/2), floor(Int,(size(h,2)-1)/2)
		pad1 = (padu, padd, padl, padr)
		pad2 = (padd, padu, padr, padl)
		# cdims reference being kept, rename variable to cdims2
		cdims2 = DenseConvDims(NNlib.pad_circular(x, pad1), h)
		cdims2ᵀ= DenseConvDims(NNlib.pad_circular(x, pad2), hᵀ)
		H = x->NNlib.conv(NNlib.pad_circular(x, pad1), h, cdims2)
		Hᵀ= x->NNlib.conv(NNlib.pad_circular(x, pad2), hᵀ,cdims2ᵀ)
	end

	for _ in 1:maxit
		x = CUDA.CUFFT.irfft(C.*(CUDA.CUFFT.rfft( Hᵀ(y) + ρ*Dᵀ(z-u), (1,2) )), M, (1,2)); # x update
		Dxᵏ = D(x);
		z = thresh_type(Dxᵏ+u, τ);            # z update
		u = u + Dxᵏ - z;                      # dual ascent
	end
	x = permutedims(x, (1,2,4,3));

	return x
end
