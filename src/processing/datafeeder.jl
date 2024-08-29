using Images, FileIO, Glob, Pipe, MLUtils
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

        if (length(x_paths) == 0) || (length(y_paths) == 0) 
            @warn "Provided paths resulted in empty list of images. X DATA: $(size(x_paths)); Y DATA: $(size(y_paths))"
        end

        return new{typeof(x_source), typeof(x_targe_shape[1])}(x_paths, y_paths, x_targe_shape, y_target_shape, shuffle)
    end
end


function get_x_y_images(data::ImageDataFeeder, idx::Integer)
    imgx = img2tensor(Images.load(data.x_data[idx]))
    imgy = img2tensor(Images.load(data.y_data[idx]))
    imsize = size(imgy)

    if (data.y_shape[1] > imsize[1]) || (data.y_shape[2] > imsize[2])
        @warn "Desired target shape $(data.y_shape) is greater than the maximum size of the target image $imsize. Complete image will be returned"
        
        return imgx, imgy
    end

    h_ref = rand(1:(imsize[1]-data.y_shape[1]+1))
    w_ref = rand(1:(imsize[2]-data.y_shape[2]+1))
    
    return imgx[h_ref:(h_ref+data.x_shape[1]-1), w_ref:(w_ref+data.x_shape[2]-1), :], imgy[h_ref:(h_ref+data.y_shape[1]-1), w_ref:(w_ref+data.y_shape[2]-1), :]
end


function Base.length(dataset::ImageDataFeeder)
    return size(dataset.y_data)[end]
end


function Base.getindex(dataset::ImageDataFeeder, idxs::Union{UnitRange, Vector, Integer})
    if typeof(idxs) <: Integer
        idxs = Vector{Integer}([idxs])
    end
    
    batches_xy = map(idx -> get_x_y_images(dataset, idx), idxs)

    batch_x = map(idx -> first(batches_xy[idx]), 1:length(batches_xy))
    batch_y = map(idx -> last(batches_xy[idx]), 1:length(batches_xy))

    batch_x = @pipe cat(batch_x..., dims=ndims(batch_x[end])+1)
    batch_y = @pipe cat(batch_y..., dims=ndims(batch_y[end])+1)

    return batch_x, batch_y
end


function MLUtils.numobs(data::ImageDataFeeder)
    return length(data)
end


function MLUtils.getobs(data::ImageDataFeeder, idxs::Union{UnitRange, Vector, Integer})
    return getindex(data, idxs)
end
