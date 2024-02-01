using CUDA, NNlib, NNlibCUDA, MLUtils, Statistics, ChainRulesCore
include("../utilities/base_funcs.jl")


# Gaussian kernel std=1.5, length=11
const SSIM_KERNEL = 
    [0.00102838008447911,
    0.007598758135239185,
    0.03600077212843083,
    0.10936068950970002,
    0.2130055377112537,
    0.26601172486179436,
    0.2130055377112537,
    0.10936068950970002,
    0.03600077212843083,
    0.007598758135239185,
    0.00102838008447911]

"""
    ssim_kernel(T, N)

Return Gaussian kernel with σ=1.5 and side-length 11 for use in [`ssim`](@ref).
Returned kernel will be `N-2` dimensional of type `T`.
"""
function ssim_kernel(T::Type, N::Integer)
    if N-2 == 1
        kernel = SSIM_KERNEL
    elseif N-2 == 2
        kernel = SSIM_KERNEL*SSIM_KERNEL' 
    elseif N-2 == 3
        ks = length(SSIM_KERNEL)
        kernel = reshape(SSIM_KERNEL*SSIM_KERNEL', 1, ks, ks).*SSIM_KERNEL
    else
        throw("SSIM is only implemented for 3D/4D/5D inputs, dimension=$N provided.")
    end
    return reshape(T.(kernel), size(kernel)..., 1, 1)
end
ChainRulesCore.@non_differentiable ssim_kernel(T::Any, N::Any)

"""
    ssim_kernel(x::AbstractArray{T, N}) where {T, N}

Return Gaussian kernel with σ=1.5 and side-length 11 for use in [`ssim`](@ref). 
Returned array will be on the same device as `x`.
"""
ssim_kernel(x::CGPUArray{T, N}) where {T, N} = ssim_kernel(T, N)
ChainRulesCore.@non_differentiable ssim_kernel(x::Any)

function _check_sizes(x::CGPUArray, y::CGPUArray)
    for d in 1:max(ndims(x), ndims(y)) 
        size(x,d) == size(y,d) || throw(DimensionMismatch(
          "loss function expects size(ŷ) = $(size(ŷ)) to match size(y) = $(size(y))"
        ))
    end
end
_check_sizes(ŷ, y) = nothing  # pass-through, for constant label e.g. y = 1
ChainRulesCore.@non_differentiable _check_sizes(ŷ::Any, y::Any)


"""
    ssim(x, y, kernel=ssim_kernel(x); peakval=1, crop=true, dims=:)
                                    
Return the [structural similarity index
measure](https://en.wikipedia.org/wiki/Structural_similarity) (SSIM) between
two signals. SSIM is computed via the mean of a sliding window of
statistics computed between the two signals. By default, the sliding window is
a Gaussian with side-length 11 in each signal dimension and σ=1.5. `crop=false` will pad `x` and `y` 
such that the sliding window computes statistics centered at every pixel of the input (via same-size convolution). 
`ssim` computes statistics independently over channel and batch dimensions.
`x` and `y` may be 3D/4D/5D tensors with channel and batch-dimensions.

`peakval=1` is the standard for image comparisons, but in practice should be
set to the maximum value of your signal type. 

`dims` determines which dimensions to average the computed statistics over. If
`dims=1:ndims(x)-1`, SSIM will be computed for each batch-element separately.

The results of `ssim` are matched against those of
[ImageQualityIndexes](https://github.com/JuliaImages/ImageQualityIndexes.jl)
for grayscale and RGB images (i.e. x, y both of size (N1, N2, 1, B) and (N1, N2, 3, B) for grayscale and color images, resp.).

See also [`ssim_loss`](@ref), [`ssim_loss_fast`](@ref).
"""
function ssim(x::CGPUArray{T,N}, y::CGPUArray{T,N}, kernel_ref=ssim_kernel(x); peakval=T(1.0), crop=true, dims=:) where {T,N}
    _check_sizes(x, y)

    if typeof(x) <: CuArray
        kernel = CUDA.cu(kernel_ref)
    else
        kernel = kernel_ref
    end

    # apply same kernel on each channel dimension separately via groups=in_channels
    groups = CUDA.size(x, N-1)
    if CUDA.size(kernel, N) != groups
        kernel = CUDA.repeat(kernel, 1,1,1,groups)
    end

    # constants to avoid division by zero
    SSIM_K = (0.01, 0.03) 
    C₁, C₂ = @. T(peakval * SSIM_K)^2

    # crop==true -> valid-sized conv (do nothing), 
    # otherwise, pad for same-sized conv
    if !crop
        # from Flux.jl:src/layers/conv.jl (calc_padding)
        padding = Tuple(mapfoldl(i -> [cld(i, 2), fld(i,2)], vcat, size(kernel)[1:N-2] .- 1))
        x = NNlibCUDA.pad_symmetric(x, padding) 
        y = NNlibCUDA.pad_symmetric(y, padding) 
    end

    μx  = conv(x, kernel, groups=groups)
    μy  = conv(y, kernel, groups=groups)
    μx² = μx.^2
    μy² = μy.^2
    μxy = μx.*μy
    σx² = conv(x.^2, kernel, groups=groups) .- μx²
    σy² = conv(y.^2, kernel, groups=groups) .- μy²
    σxy = conv(x.*y, kernel, groups=groups) .- μxy

    ssim_map = @. (2μxy + C₁)*(2σxy + C₂)/((μx² + μy² + C₁)*(σx² + σy² + C₂))
    return mean(ssim_map, dims=dims)
end

"""
    ssim_loss(x, y, kernel=ssim_kernel(x); peakval=1, crop=true, dims=:)

Computes `1 - ssim(x, y)`, suitable for use as a loss function with gradient descent.
For faster training, it is recommended to store a kernel and reuse it, ex.,
```julia
kernel = ssim_kernel(Float32, 2) |> gpu
# or alternatively for faster computation
# kernel = ones(Float32, 5, 5, 1, num_channels) |> gpu

for (x, y) in dataloader
    x, y = (x, y) .|> gpu
    grads = gradient(model) do m
        x̂ = m(y)
        ssim_loss(x, x̂, kernel)
    end
    # update the model ...
end
```
See [`ssim`](@ref) for a detailed description of SSIM and the above arguments.
See also [`ssim_loss_fast`](@ref).
"""
ssim_loss(x::CGPUArray{T}, args...; kws...) where T = one(T) - ssim(x, args...; kws...)

"""
    ssim_loss_fast(x, y; kernel_length=5, peakval=1, crop=true, dims=:)

Computes `ssim_loss` with an averaging kernel instead of a large Gaussian
kernel for faster computation. `kernel_length` specifies the averaging kernel
side-length in each signal dimension of x, y.  See [`ssim`](@ref) for a
detailed description of SSIM and the above arguments. 

See also [`ssim_loss`](@ref).
"""
function ssim_loss_fast(x::CGPUArray{T, N}, y::CGPUArray{T, N}; kernel_length=5, kws...) where {T, N}
    kernel = ones_like(x, (kernel_length*ones(Int, N-2)..., 1, size(x, N-1)))
    kernel = kernel ./ sum(kernel; dims=1:N-1)
    return ssim_loss(x, y, kernel; kws...)
end


function peak_snr(x::CGPUArray, y::CGPUArray, peak_val::Number=1.0f0)
    mse = mean((y .- x).^2.0f0)
    if mse == 0.0f0
        return 100
    end
    return 20.0f0 .* log10(peak_val ./ sqrt(mse))
end