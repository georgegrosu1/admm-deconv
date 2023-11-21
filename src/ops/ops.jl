using FFTW, DSP, ImageFiltering, Flux
using Images, ColorVectorSpace, ColorTypes

include("../utils.jl")



Base.@kwdef mutable struct ADMMSettings 
    rho_r::Float32 = 2.0
    rho_o::Float32 = 50.0
    beta::Vector{Float32} = [1.f0, 1.f0, 2.5f0]
    gamma::Float32 = 2.0
    max_iter::UInt32 = 20
    alpha::Float32 = 0.7
    tolerance::Float32 = 1e-3
end


function admm_tv_deconv(input::Array{Float32}, 
                        kern::Array{Float32}, 
                        regularize_val::Float32, 
                        admmSettings::ADMMSettings) 

    input_size = size(input)

    if ndims(kern) < ndims(input)
        kern = expand_dims(kern, ndims(kern) + 1)
    end

    bandpass_kern = [1 -1]
    bandpass_kern = expand_dims(bandpass_kern, ndims(bandpass_kern) + 1)
    bandpass_kern_ch = expand_dims(bandpass_kern, 1)
    
    eigHtH = abs.(fftnMatLike(kern, input_size)).^2
    eigDtD = abs.(admmSettings.beta[1] .* fftnMatLike(bandpass_kern, input_size)).^2 + abs.(admmSettings.beta[2] .* fftnMatLike(collect(bandpass_kern'), input_size)).^2
    eigEtE = input_size[end] > 1 ? abs.(admmSettings.beta[3] .* fftnMatLike(bandpass_kern_ch, input_size)).^2 : 0;

    Htg = imfilter(input, kern, [border="circular"])

    
end