using FFTW, DSP, ImageFiltering, Flux
using Images, ColorVectorSpace, ColorTypes

include("../utilities/base_funcs.jl")



Base.@kwdef mutable struct ADMMSettings 
    rho_r::Float32 = 2.0
    rho_o::Float32 = 50.0
    beta::Vector{Float32} = [1.f0, 1.f0, 2.5f0]
    gamma::Float32 = 2.0
    max_iter::UInt32 = 20
    alpha::Float32 = 0.7
    lGrange_M1::Array{Float32} = Array{Float32}(undef)
    lGrange_M2::Array{Float32} = Array{Float32}(undef)
    lGrange_M3::Array{Float32} = Array{Float32}(undef)
    z::Array{Float32} = Array{Float32}(undef)
end


function admm_tv_deconv(input::Array{Float32, 4}, 
                        kern::Array{Float32, 4}, 
                        num_iters::Int32,
                        regularize_val::Float32, 
                        cfgs::ADMMSettings)::Array{Float32, 4}

    input_size = size(input)

    cfgs.lGrange_M1 = zeros(Float32, input_size)
    cfgs.lGrange_M2 = zeros(Float32, input_size)
    cfgs.lGrange_M3 = zeros(Float32, input_size)
    cfgs.z = zeros(Float32, input_size)

    restored = input

    if ndims(kern) < ndims(input)
        kern = expand_dims(kern, ndims(kern) + 1)
    end

    bandpass_kern_x = expand_dims(expand_dims(expand_dims(Float32.([1, -1]), 2), 3), 4)
    bandpass_kern_y = expand_dims(expand_dims(Float32.([1 -1]), 3), 4)
    bandpass_kern_z = expand_dims(expand_dims(Float32.([1 -1]), 3), 1)
    
    eigHtH = abs.(fftnMatLike(kern, input_size)).^2
    eigDtD = abs.(cfgs.beta[1] .* fftnMatLike(bandpass_kern_x, input_size)).^2 + abs.(cfgs.beta[2] .* fftnMatLike(collect(bandpass_kern_y), input_size)).^2
    eigEtE = input_size[end] > 1 ? abs.(cfgs.beta[3] .* fftnMatLike(bandpass_kern_z, input_size)).^2 : 0;

    Htg = imfilter(input, kern, "circular")

    Δx, Δy, Δz = forward_diff3d(input)
    Γ = imfilter(restored, kern, "circular") - input
    rNorm = sqrt(norm(Δx)^2 + norm(Δy)^2 + norm(Δz)^2)

    for _ in 1:num_iters
    
        # Perform the U-subproblem
        v_Δx = Δx .+ (1 / cfgs.rho_r) .* cfgs.lGrange_M1
        v_Δy = Δy .+ (1 / cfgs.rho_r) .* cfgs.lGrange_M2
        v_Δz = Δz .+ (1 / cfgs.rho_r) .* cfgs.lGrange_M3
        u_sqSum = sqrt.(v_Δx.^2 + v_Δy.^2 + v_Δz.^2)
        u_sqSum[u_sqSum .== 0] .= nextfloat(typemin(Float32))
        u_sqSum = max.(u_sqSum .- 1 / cfgs.rho_r) ./ u_sqSum

        u_Δx = v_Δx .* u_sqSum
        u_Δy = v_Δy .* u_sqSum
        u_Δz = v_Δz .* u_sqSum

        # Perform the R-subproblem
        r_Δ = max.(abs.(Γ .+ 1 / cfgs.rho_o .* cfgs.z) .- regularize_val / cfgs.rho_o, 0) .* sign.(Γ .+ 1 / cfgs.rho_o * cfgs.z)
        
        # Perform the F-subproblem
        rhs = cfgs.rho_o .* Htg .+ imfilter(cfgs.rho_o .* r_Δ, kern, "circular") + divergence3d(cfgs.rho_r .* u_Δx .- cfgs.lGrange_M1, cfgs.rho_r .* u_Δy .- cfgs.lGrange_M2, cfgs.rho_r .* u_Δz .- cfgs.lGrange_M3)
        eigA = cfgs.rho_o .* eigHtH .+ cfgs.rho_r .* eigDtD .+ cfgs.rho_r .* eigEtE
        restored = real.(ifft(fft(rhs) ./ eigA))

        # Update process parameters forward diffs, Lagrange multipliers, Gamma, z, rNorm
        Δx, Δy, Δz = forward_diff3d(restored)
        Γ = imfilter(restored, kern, "circular") - input

        cfgs.lGrange_M1 -= cfgs.rho_r .* (u_Δx .- Δx)
        cfgs.lGrange_M2 -= cfgs.rho_r .* (u_Δy .- Δy)
        cfgs.lGrange_M3 -= cfgs.rho_r .* (u_Δz .- Δz)
        cfgs.z .-= cfgs.rho_o .* (r_Δ .- Γ)

        prev_rNorm = rNorm
        rNorm = sqrt(norm(Δx .- u_Δx, 2)^2 + norm(Δy .- u_Δy, 2)^2 + norm(Δz .- u_Δz, 2)^2)

        if rNorm > (cfgs.alpha * prev_rNorm)
            cfgs.rho_r = cfgs.rho_r * cfgs.gamma
        end
    end

    return restored
end