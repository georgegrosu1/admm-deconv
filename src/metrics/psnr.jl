using Flux, CUDA
include("../utilities/base_funcs.jl")


function peak_snr(x::CGPUArray, y::CGPUArray, peak_val::Number=1.0f0)
    mse = Flux.mean((y .- x).^2.0f0, dims=[1, 2, 3])
    if mse == 0.0f0
        return 100f0
    end
    return Flux.mean(20.0f0 .* log10.(peak_val ./ sqrt.(mse)))
end