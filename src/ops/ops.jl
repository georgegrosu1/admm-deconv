using FFTW, DSP, ImageFiltering, NNlib, Flux
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
	W = zeros(T, 2,2,1,2)
	W[:,:,1,1] = [1 -1; 0 0]; 
	W[:,:,1,2] = [1  0;-1 0]; 
	Wᵀ = reverse(permutedims(W, (2,1,4,3)), dims=:);
	return W, Wᵀ
end


function tvd_fft(y::AbstractArray, λ, ρ=1; h=missing, isotropic=false, maxit=100, tol=1e-2, verbose=true) 
	M, N, P, _ = size(y)
	y = permutedims(y, (1,2,4,3)) # move channels to batch dimension
	τ = λ / ρ;

	if !ismissing(h)
		hh = NNlib.pad_constant(h, ((0, M-size(h)[1]), (0, N-size(h)[2])))
		Σ = rfft(hh)
	else
		Σ = 1
	end

	# precompute C for x-update
	Λx = rfft([1 -1 zeros(N-2)'; zeros(M-1,N)]);
	Λy = rfft([[1; -1; zeros(M-2)] zeros(M,N-1)])
	C = 1 ./ ( abs2.(Σ) .+ ρ.*(abs2.(Λx) .+ abs2.(Λy)) )

	# real Fourier xfrm in image dimension.
	# Must specify length of first dimension for inverse.
	Q  = plan_rfft(y,(1,2)); 
	Qᴴ = plan_irfft(rfft(y),M,(1,2));

	if isotropic
		objfun = (x,Dx) -> objfun_iso(x,Dx,y,λ)
		T = BT # block-thresholding
	else
		objfun = (x,Dx) -> objfun_aniso(x,Dx,y,λ)
		T = ST # soft-thresholding
	end

	# initialization
	x = zeros(M,N,1,P);
	Dxᵏ = zeros(M,N,2,P);
	z = zeros(M,N,2,P);
	u = zeros(M,N,2,P);

	# conv kernel
	W, Wᵀ = fdkernel(eltype(y))

	# (in-place) Circular convolution
	cdims = DenseConvDims(NNlib.pad_circular(x, (1,0,1,0)), W);
	cdimsᵀ= DenseConvDims(NNlib.pad_circular(z, (0,1,0,1)), Wᵀ);
	D!(z,x) = NNlib.conv!(z, NNlib.pad_circular(x, (1,0,1,0)), W, cdims);
	Dᵀ!(x,z)= NNlib.conv!(x, NNlib.pad_circular(z, (0,1,0,1)), Wᵀ,cdimsᵀ);
	D(x) = NNlib.conv(NNlib.pad_circular(x, (1,0,1,0)), W, cdims);
	Dᵀ(z) = NNlib.conv(NNlib.pad_circular(z, (0,1,0,1)), Wᵀ,cdimsᵀ);

	if !ismissing(h)
		h = h[:,:,:,:]
		hᵀ= rot180(h[:,:])[:,:,:,:]
		padu, padd = ceil(Int,(size(h,1)-1)/2), floor(Int,(size(h,1)-1)/2)
		padl, padr = ceil(Int,(size(h,2)-1)/2), floor(Int,(size(h,2)-1)/2)
		pad1 = (padu, padd, padl, padr)
		pad2 = (padd, padu, padr, padl)
		# cdims reference being kept, rename variable to cdims2
		cdims2 = DenseConvDims(NNlib.pad_circular(x, pad1), h);
		cdims2ᵀ= DenseConvDims(NNlib.pad_circular(x, pad2), hᵀ);
		H = x->NNlib.conv(NNlib.pad_circular(x, pad1), h, cdims2);
		Hᵀ= x->NNlib.conv(NNlib.pad_circular(x, pad2), hᵀ,cdims2ᵀ);
	else
		H = identity
		Hᵀ= identity
	end

	for _ in 1:maxit
		x = Qᴴ*(C.*(Q*( Hᵀ(y) + ρ*Dᵀ(z-u) ))); # x update
		D!(Dxᵏ,x);
		z = T(Dxᵏ+u, τ);                      # z update
		u = u + Dxᵏ - z;                      # dual ascent
	end
	x = permutedims(x, (1,2,4,3));

	return x
end
