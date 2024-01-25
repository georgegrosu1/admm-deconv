using Images, Flux, FileIO, Glob, Pipe
include("../utilities/base_funcs.jl")


mutable struct ImageDataFeeder{T, M}
    x_data::Vector{T}
    y_data::Vector{T}
    x_shape::Tuple{M, M}
    y_shape::Tuple{M, M}
    shuffle::Bool

    function ImageDataFeeder(x_source::AbstractString = "", 
                             y_source::AbstractString = "", 
                             extension::String = ".png",
                             x_targe_shape::Tuple{Integer, Integer} = (),
                             y_target_shape::Tuple{Integer, Integer} = (),
                             shuffle::Bool = false)
        
        x_paths = Glob.glob("*" * extension, x_source)
        y_paths = Glob.glob("*" * extension, y_source)

        return new{typeof(x_source), typeof(x_targe_shape[1])}(x_paths, y_paths, x_targe_shape, y_target_shape, shuffle)
    end
end


function get_image(data::Vector, idx::Integer, target_shape::Tuple{Integer, Integer})
    img = img2tensor(Images.load(data[idx]))
    imsize = size(img)

    if (target_shape[1] > imsize[1]) || (target_shape[2] > imsize[2])
        @warn "Desired target shape $target_shape is greater than the maximum size of the target image $imsize. Complete image will be returned"
        
        return img
    end
    
    h_ref = rand(1:(imsize[1]-target_shape[1]+1))
    w_ref = rand(1:(imsize[2]-target_shape[2]+1))

    return img[h_ref:(h_ref+target_shape[1]-1), w_ref:(w_ref+target_shape[2]-1), :]
end


function Base.length(dataset::ImageDataFeeder)
    return length(dataset.y_data)
end


function Base.getindex(dataset::ImageDataFeeder, idxs::Union{UnitRange, Vector, Integer})
    if typeof(idxs) <: Integer
        idxs = Vector{Integer}([idxs])
    end
    
    batch_in = map(idx -> get_image(dataset.x_data, idx, dataset.x_shape), idxs)
    batch_gt = map(idx -> get_image(dataset.y_data, idx, dataset.y_shape), idxs)

    batch_x = @pipe cat(batch_in..., dims=ndims(batch_in[end])+1)
    batch_y = @pipe cat(batch_gt..., dims=ndims(batch_gt[end])+1)

    return batch_x, batch_y
end
