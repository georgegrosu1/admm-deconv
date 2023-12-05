using Images, Flux, FileIO, Glob
include("../utilities/base_funcs.jl")


mutable struct ImageDataFeeder{T, M}
    x_data::Vector{T}
    y_data::Vector{T}
    x_shape::Tuple{M, M}
    y_shape::Tuple{M, M}
    batch_size::Int32
    shuffle::bool

    function ImageDataFeeder(x_source::AbstractString = "", 
                             y_source::AbstractString = "", 
                             extension::String = ".png",
                             x_targe_shape::Tuple{Integer, Integer} = (),
                             y_target_shape::Tuple{Integer, Integer} = (),
                             batch_size::Integer = 1,
                             shuffle::bool = false)
        
        x_paths = Glob.glob("*" * extension, x_source)
        y_paths = Glob.glob("*" * extension, y_source)

        return new{typeof(x_source), typeof(x_targe_shape[1])}(x_paths, y_paths, x_targe_shape, y_target_shape, batch_size, shuffle)
    end
end


function Base.length(dataset::ImageDataFeeder)
    return length(dataset.y_data)
end


function Base.getindex(dataset::ImageDataFeeder, idxs::Union{UnitRange,Vector})
    batch_in = map(idx -> Images.load(dataset.x_data[idx]), idxs)
    batch_gt = map(idx -> Images.load(dataset.y_data[idx]), idxs)

    return batch_in, batch_gt
end
