using ImageFiltering, Images, ColorVectorSpace, ColorTypes, FFTW, Flux, DSP


function expand_dims(arr::Array, dim_idx::Int64)
    s = [size(arr)...]
    insert!(s, dim_idx, 1)
    return reshape(arr, s...)
end


function UInt8RGB2arr(rgb_img::Matrix{RGB{N0f8}})
    permutedims(channelview(rgb_img), [2, 3, 1])
end


function UInt8Gray2arr(gray_img::Matrix{Gray{N0f8}})
    x = permutedims(channelview(gray_img), [1, 2])
    expand_dims(x, ndims(x)+1)
end


function img2tensor(img_in::Matrix)
    im_out = Float32.(reinterpret(reshape, N0f8, img_in) |> collect)
    if ndims(im_out) == 3
        im_out = permutedims(im_out, (2, 3, 1))
    end
    return im_out
end


function fftnMatLike(input_arr::Array, out_shape::Tuple)::Array{ComplexF32}
    @assert ndims(input_arr) == length(out_shape) "out_shape must be of size equal to the number of dimensions of input!"

    input_size = size(input_arr)
    pads_vec::Vector = []
    for i=eachindex(out_shape)
        push!(pads_vec, (0, out_shape[i] - input_size[i]))
    end
    pads = Tuple(pads_vec)

    return fft(pad_constant(input_arr, pads, 0))
end


function forward_diff3d(data::Array{Float32, 4}, beta::Vector{Float32}=[1f0, 1f0, 1f0])::Tuple{Array{Float32,4}, Array{Float32, 4}, Array{Float32, 4}} 
    @assert length(beta) == 3 "beta param. must have 3 elements"

    Δx = diff(data, dims=1)
    Δx_resid = expand_dims(data[begin, :, :, :] - data[end, :, :, :], 1)
    Δx = beta[1] * cat(Δx, Δx_resid, dims=1)

    Δy = diff(data, dims=2)
    Δy_resid = expand_dims(data[:, begin, :, :] - data[:, end, :, :], 2)
    Δy = beta[2] * cat(Δy, Δy_resid, dims=2)

    Δz = diff(data, dims=3)
    Δz_resid = expand_dims(data[:, :, begin, :] - data[:, :, end, :], 3)
    Δz = beta[3] * cat(Δz, Δz_resid, dims=3)

    return Δx, Δy, Δz
end


function divergence3d(x::Array{Float32, 4}, 
                      y::Array{Float32, 4}, 
                      z::Array{Float32, 4}, 
                      beta::Vector{Float32}=[1f0, 1f0, 0f0])::Array{Float32, 4}

    @assert length(beta) == 3 "beta param. must have 3 elements"

    Δdim = -diff(x, dims=1)
    Δdim_resid = expand_dims(x[begin, :, :, :] - x[end, :, :, :], 1)
    div = beta[1] * cat(Δdim_resid, Δdim, dims=1)

    Δdim = -diff(y, dims=2)
    Δdim_resid = expand_dims(y[:, end, :, :] - y[:, begin, :, :], 2)
    div += beta[2] * cat(Δdim_resid, Δdim, dims=2)

    Δdim = -diff(z, dims=3)
    Δdim_resid = expand_dims(z[:, :, end, :] - y[:, :, begin, :], 3)
    div += beta[3] * cat(Δdim_resid, Δdim, dims=3)

    return div
end
