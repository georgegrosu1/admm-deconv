using Flux, CUDA, ChainRulesCore
include("../utilities/base_funcs.jl")


const PREWITT_KERNEL_X = cat(
    [1.f0, 1.f0, 1.f0], 
    [0.f0, 0.f0, 0.f0],
    [-1.f0, -1.f0, -1.f0], dims=2
) ./ 3.f0
const PREWITT_KERNEL_Y = collect(PREWITT_KERNEL_X')

const SOBEL_KERNEL_X = cat(
    [1.f0, 0.f0, -1.f0], 
    [2.f0, 0.f0, -2.f0],
    [1.f0, 0.f0, -1.f0], dims=2
) ./ 8.f0
const SOBEL_KERNEL_Y = collect(SOBEL_KERNEL_X')

ChainRulesCore.@non_differentiable PREWITT_KERNEL_X(x::Any)
ChainRulesCore.@non_differentiable PREWITT_KERNEL_Y(x::Any)
ChainRulesCore.@non_differentiable SOBEL_KERNEL_X(x::Any)
ChainRulesCore.@non_differentiable SOBEL_KERNEL_Y(x::Any)

function imgrads(x::CGPUArray{T, N}) where {T,N}
    groups = size(x, N-1)
    
    if typeof(x) <: CuArray
        ker_x = cu(SOBEL_KERNEL_X)
        ker_y = cu(SOBEL_KERNEL_Y)
    else
        ker_x = SOBEL_KERNEL_X
        ker_y = SOBEL_KERNEL_Y
    end

    
    if size(ker_x, N) != groups
        kernel_x = repeat(ker_x, 1,1,1,groups)
        kernel_y = repeat(ker_y, 1,1,1,groups)
    else
        kernel_x = ker_x
        kernel_y = ker_y
    end

    padding = Int((CUDA.size(kernel_x, 1)-1)/2)
    
    xgrad = conv(pad_circular(x, padding), kernel_x, groups=groups)
    ygrad = conv(pad_circular(x, padding), kernel_y, groups=groups)

    return (xgrad, ygrad)
end


function gradientsmag(gradx::CGPUArray, grady::CGPUArray)
    return sqrt.(gradx.^2 + grady.^2 .+ 1f-16)
end
