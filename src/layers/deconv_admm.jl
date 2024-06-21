using Flux, CUDA
include("../ops/ops.jl")


#------------------------------------------------------------------------------------------------------------------------------
mutable struct ADMMDeconv{F,A,N,V,M,B,C,D}
  σ::F
  weight::A
  bias::V
  λ::N
  ρ::M
  iters::B
  iso::C
  creg::D
end


function ADMMDeconv(w::AbstractArray{T,N}, 
                    σ,
                    b,
                    lambda,
                    rho,
                    iters,
                    iso,
                    creg) where {T,N}

  return ADMMDeconv(σ, w, b, lambda, rho, iters, iso, creg)
end


function ADMMDeconv(k::NTuple{N,Integer},
                    num_it::Integer,
                    σ = Flux.identity; 
                    iso::Bool = false,
                    init = Flux.glorot_uniform, 
                    groups = 1,
                    bias = false,
                    creg::Number = 0f0) where {N}

  weight = Flux.convfilter(k, 1=>1; init=init, groups=groups)
  λ = abs.(Flux.glorot_uniform(1))
  ρ = abs.(Flux.glorot_uniform(1))
  bias_w = Flux.create_bias(weight, bias, 1)
  ADMMDeconv(weight, σ, bias_w, λ, ρ, num_it, iso, creg)
end


Flux.@functor ADMMDeconv weight, bias, λ, ρ


function (d::ADMMDeconv)(x::AbstractArray)
  # d.λ = clamp.(d.λ, d.creg, Inf32)
  # d.ρ = clamp.(d.ρ, d.creg, Inf32)

  # d.weight = clamp.(d.weight, 0f0, Inf32)
  
  res = tvd_fft(x, d.λ, d.ρ, d.weight, d.iso, d.iters)
  res = res .+ d.bias
  
  return d.σ.(res)
end
#------------------------------------------------------------------------------------------------------------------------------
