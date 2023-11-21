using ImageFiltering, Images, ColorVectorSpace, ColorTypes, FFTW, Flux, DSP


function expand_dims(arr::Array, dim_id::Int64)
    s = [size(arr)...]
    insert!(s, dim_id, 1)
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


function forward_diff3d(data::Array{Float32, 4}, beta::Vector{Float32}=[Float32(1), Float32(1), Float32(0)])
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
                      beta::Vector{Float32}=[Float32(1), Float32(1), Float32(0)])

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


function convolve(input_arr::Array{T}, kernel::Array{T}, mode::String="full")
    @assert ndims(input_arr) == ndims(kernel) "Input and kernel must have same number of dimensions"
    conv_out = DSP.conv(input_arr, kernel)
    if mode == "full"
        return conv_out
    elseif mode == "same"
        start_end_idxs = ()
        input_size = size(input_arr)
        kern_size = size(kernel)
        out_size = size(conv_out)
        for i=1:ndims(conv_out)
            out_dim_diff = out_size[i] - max(input_size[i], kern_size[i])
            cut_left = out_dim_diff % 2 == 0 ? div(out_dim_diff, 2) : div(out_dim_diff, 2) + 1
            cut_right = div(out_dim_diff, 2)
            start_end_idxs = (start_end_idxs..., (cut_left, cut_right))
        end
    end

end
