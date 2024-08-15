using Flux, CUDA
include("iqa_utils.jl")


function similarity_map(map_xref::CGPUArray, map_x::CGPUArray, constant::Float32, α::Float32)
    num = @. 2.f0 * map_xref * map_x - α * map_xref * map_x + constant
    denum = @. map_xref^2 + map_x^2 - α * map_xref * map_x + constant

    return num ./ denum
end


function gmsd(x::CGPUArray{T,N}, y::CGPUArray{T,N}, t::Float32=0.0026f0, α::Float32=0.f0, reduction::Function=Flux.mean) where {T,N}
    x_gradx, x_grady = imgrads(x)
    y_gradx, y_grady = imgrads(y)

    map_x = gradientsmag(x_gradx, x_grady)
    map_y = gradientsmag(y_gradx, y_grady)

    gms = similarity_map(map_x, map_y, t, α)

    mean_gms = Flux.mean(gms, dims=[1, 2, 3])

    score = Flux.mean((gms .- mean_gms).^2, dims=[1, 2, 3])

    return reduction(sqrt.(score))
end


gmsd_loss(x::CGPUArray{T}, args...; kws...) where T = gmsd(x, args...; kws...)