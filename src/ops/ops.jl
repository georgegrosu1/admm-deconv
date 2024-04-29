using  Flux, CUDA, NNlib, NNlibCUDA, NNlib, FFTW

include("../utilities/base_funcs.jl")


pixelnorm(x) = sqrt.(sum(x.^2, dims=(3,4))) # 2-norm on 4D image-tensor pixel-vectors

HT(x,τ) = x*(abs(x) > τ)                      # hard-thresholding
ST(x,τ) = sign.(x).*max.(abs.(x).-τ, 0f0)       # soft-thresholding
BT(x,τ) = max.(1 .- τ ./ pixelnorm(x), 0).*x   # block-thresholding   # block-thresholding
GT(x, τ) = (exp.(-pixelnorm(x).^2f0 ./ (2f0 .* τ.^2f0)) .* 0.5f0 .* -1f0 .+ 0.5f0) .* x # gaussian-thresholding

objfun_iso(x,Dx,y,λ)   = 0.5f0*sum(abs2.(x-y)) + λ*norm(pixelnorm(Dx), 1) 
objfun_aniso(x,Dx,y,λ) = 0.5f0*sum(abs2.(x-y)) + λ*norm(Dx, 1)


function tvd_fft_cpu(y::AbstractArray{T}, λ, ρ::AbstractArray{T}=[1], h::AbstractArray{T}=[], isotropic=false, maxit=100) where {T}
	M, N, P, B = size(y)
	y = permutedims(y, (1,2,4,3)) # move channels to batch dimension
	τ = λ ./ ρ

	if Flux.isempty(h)
		Σ = AbstractArray{T}([1])
	else
		hh = NNlib.pad_constant(h, (0, M-size(h)[1], 0, N-size(h)[2], 0, 0, 0, 0))
		Σ_ref = rfft(dropdims(hh, dims=(3,4)))
		Σ = reshape(Σ_ref, size(Σ_ref)..., 1, 1)
	end

	# precompute C for x-update
	# Compose Dx filter
	dx_filter = cat(cat(AbstractArray{T}([1 -1]), zeros(T, (1,N-2)), dims=2), zeros(T, (M-1,N)), dims=1)
	# Compose Dy filter
	dy_filter = cat(cat(AbstractArray{T}([1; -1]), zeros(T, (M-2)), dims=1), zeros(T, (M,N-1)), dims=2)
	Λx = rfft(dx_filter)
	Λy = rfft(dy_filter)
	C = 1 ./ ( abs2.(Σ) .+ ρ.*(abs2.(Λx) .+ abs2.(Λy)) )

	if isotropic
		thresh_type = BT # block-thresholding
	else
		thresh_type = ST # soft-thresholding
	end

	# initialization
	x::AbstractArray{T} = zeros(T, M,N,B,P)
	Dxᵏ::AbstractArray{T} = zeros(T, M,N,2*B,P)
	z::AbstractArray{T} = zeros(T, M,N,2*B,P)
	u::AbstractArray{T} = zeros(T, M,N,2*B,P)

	# W conv kernel
	W1 = cat(cat(ones(T, 1,1), -ones(T, 1,1), dims=2), zeros(T, 1,2), dims=1)
	W2 = cat(cat(ones(T, 1), -ones(T, 1), dims=1), zeros(T, 2), dims=2)
	W = repeat(cat(W1, W2, dims=4), 1,1,1,B)
	# Wᵀ conv kernel
	Wt1 = cat(zeros(T, 1,2), cat(-ones(T, 1,1), ones(T, 1,1), dims=2), dims=1)
	Wt2 = cat(zeros(T, 2), cat(-ones(T, 1), ones(T, 1), dims=1), dims=2)
	Wᵀ = permutedims(cat(Wt1, Wt2, dims=4), (1,2,4,3))
	Wᵀ = repeat(Wᵀ, 1,1,1,B)

	# (in-place) Circular convolution
	cdims = DenseConvDims(NNlib.pad_circular(x, (1,0,1,0)), W, groups=B)
	cdimsᵀ= DenseConvDims(NNlib.pad_circular(z, (0,1,0,1)), Wᵀ, groups=B)
	D(x) = NNlib.conv(NNlib.pad_circular(x, (1,0,1,0)), W, cdims)
	Dᵀ(z) = NNlib.conv(NNlib.pad_circular(z, (0,1,0,1)), Wᵀ,cdimsᵀ)

	if isempty(h)
		H = identity
		Hᵀ= identity
	else
		h = repeat(h, 1,1,1,B)
		hᵀ= reverse(h)
		padu, padd = ceil(Int,(size(h,1)-1)/2), floor(Int,(size(h,1)-1)/2)
		padl, padr = ceil(Int,(size(h,2)-1)/2), floor(Int,(size(h,2)-1)/2)
		pad1 = (padu, padd, padl, padr)
		pad2 = (padd, padu, padr, padl)
		# cdims reference being kept, rename variable to cdims2
		cdims2 = DenseConvDims(NNlib.pad_circular(x, pad1), h, groups=B)
		cdims2ᵀ= DenseConvDims(NNlib.pad_circular(x, pad2), hᵀ, groups=B)
		H = x->NNlib.conv(NNlib.pad_circular(x, pad1), h, cdims2)
		Hᵀ= x->NNlib.conv(NNlib.pad_circular(x, pad2), hᵀ,cdims2ᵀ)
	end

	for _ in 1:maxit
		# x update
		x = irfft(C.*(rfft( Hᵀ(y) + ρ .* Dᵀ(z-u), (1,2) )), M, (1,2))
		Dxᵏ = D(x)
		# z update
		z = thresh_type(Dxᵏ+u, τ)
		# dual ascent
		u = u + Dxᵏ - z
	end
	x = permutedims(x, (1,2,4,3))

	return x
end


function tvd_fft_gpu(y::CGPUArray{T}, λ::CGPUArray{T}, ρ::CGPUArray{T}=CuArray([1]), h::CGPUArray{T}=CuArray([]), isotropic=false, maxit=100) where {T}
	M, N, P, B = CUDA.size(y)
	y = CUDA.permutedims(y, (1,2,4,3)) # move channels to batch dimension
	τ = λ ./ ρ

	if Flux.isempty(h)
		Σ = CuArray{T}([1f0])
	else
		hh = NNlibCUDA.pad_constant(h, (0, M-CUDA.size(h)[1], 0, N-CUDA.size(h)[2], 0, 0, 0, 0))
		Σ_ref = CUDA.CUFFT.rfft(CUDA.dropdims(hh, dims=(3,4)))
		Σ = CUDA.reshape(Σ_ref, CUDA.size(Σ_ref)..., 1, 1)
	end

	# precompute C for x-update
	# Compose Dx filter
	dx_filter = CUDA.cat(CUDA.cat(CuArray{T}([1 -1]), CUDA.zeros((1,N-2)), dims=2), CUDA.zeros((M-1,N)), dims=1)
	# Compose Dy filter
	dy_filter = CUDA.cat(CUDA.cat(CuArray{T}([1; -1]), CUDA.zeros((M-2)), dims=1), CUDA.zeros((M,N-1)), dims=2)
	Λx = CUDA.CUFFT.rfft(dx_filter)
	Λy = CUDA.CUFFT.rfft(dy_filter)
	C = 1 ./ ( CUDA.abs2.(Σ) .+ ρ.*(CUDA.abs2.(Λx) .+ CUDA.abs2.(Λy)) )

	if isotropic
		thresh_type = BT # block-thresholding
	else
		thresh_type = ST # soft-thresholding
	end

	# initialization
	x::CuArray{T} = CUDA.zeros(T, M,N,B,P)
	Dxᵏ::CuArray{T} = CUDA.zeros(T, M,N,2*B,P)
	z::CuArray{T} = CUDA.zeros(T, M,N,2*B,P)
	u::CuArray{T} = CUDA.zeros(T, M,N,2*B,P)

	# W conv kernel
	W1 = CUDA.cat(CUDA.cat(CUDA.ones(1,1), -CUDA.ones(1,1), dims=2), CUDA.zeros(1,2), dims=1)
	W2 = CUDA.cat(CUDA.cat(CUDA.ones(1), -CUDA.ones(1), dims=1), CUDA.zeros(2), dims=2)
	W = CUDA.repeat(CUDA.cat(W1, W2, dims=4), 1,1,1,B)
	# Wᵀ conv kernel
	Wt1 = CUDA.cat(CUDA.zeros(1,2), CUDA.cat(-CUDA.ones(1,1), CUDA.ones(1,1), dims=2), dims=1)
	Wt2 = CUDA.cat(CUDA.zeros(2), CUDA.cat(-CUDA.ones(1), CUDA.ones(1), dims=1), dims=2)
	Wᵀ = CUDA.permutedims(CUDA.cat(Wt1, Wt2, dims=4), (1,2,4,3))
	Wᵀ = CUDA.repeat(Wᵀ, 1,1,1,B)

	# (in-place) Circular convolution
	cdims = DenseConvDims(NNlibCUDA.pad_circular(x, (1,0,1,0)), W, groups=B)
	cdimsᵀ= DenseConvDims(NNlibCUDA.pad_circular(z, (0,1,0,1)), Wᵀ, groups=B)
	D(x) = NNlibCUDA.conv(NNlibCUDA.pad_circular(x, (1,0,1,0)), W, cdims)
	Dᵀ(z) = NNlibCUDA.conv(NNlibCUDA.pad_circular(z, (0,1,0,1)), Wᵀ,cdimsᵀ)

	if isempty(h)
		H = identity
		Hᵀ= identity
	else
		h = CUDA.repeat(h, 1,1,1,B)
		hᵀ= CUDA.reverse(h)
		padu, padd = CUDA.ceil(Int,(CUDA.size(h,1)-1)/2), CUDA.floor(Int,(CUDA.size(h,1)-1)/2)
		padl, padr = CUDA.ceil(Int,(CUDA.size(h,2)-1)/2), CUDA.floor(Int,(CUDA.size(h,2)-1)/2)
		pad1 = (padu, padd, padl, padr)
		pad2 = (padd, padu, padr, padl)
		# cdims reference being kept, rename variable to cdims2
		cdims2 = DenseConvDims(NNlibCUDA.pad_circular(x, pad1), h, groups=B)
		cdims2ᵀ= DenseConvDims(NNlibCUDA.pad_circular(x, pad2), hᵀ, groups=B)
		H = x->NNlibCUDA.conv(NNlibCUDA.pad_circular(x, pad1), h, cdims2)
		Hᵀ= x->NNlibCUDA.conv(NNlibCUDA.pad_circular(x, pad2), hᵀ,cdims2ᵀ)
	end

	for _ in 1:maxit
		# x update
		x = CUDA.CUFFT.irfft(C.*(CUDA.CUFFT.rfft( Hᵀ(y) + ρ .* Dᵀ(z-u), (1,2) )), M, (1,2))
		Dxᵏ = D(x)
		# z update
		z = thresh_type(Dxᵏ+u, τ)
		# dual ascent
		u = u + Dxᵏ - z
	end
	x = CUDA.permutedims(x, (1,2,4,3))

	return x
end


function tvd_fft(y::CGPUArray{T}, λ::CGPUArray{T}, ρ::CGPUArray{T}=CuArray([1]), h::CGPUArray{T}=CuArray([]), isotropic=false, maxit=100) where {T}
	
	if typeof(y) <: CuArray
		return tvd_fft_gpu(y, λ, ρ, h, isotropic, maxit)
	end

	return tvd_fft_cpu(y, λ, ρ, h, isotropic, maxit)
end
