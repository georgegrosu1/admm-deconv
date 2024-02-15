using Flux, CUDA, NNlibCUDA
include("../ops/ops.jl")


#------------------------------------------------------------------------------------------------------------------------------
mutable struct ADMMDeconv{F,A,N,V,M,B,C}
  σ::F
  weight::A
  bias::V
  λ::N
  ρ::M
  iters::B
  iso::C
end


function ADMMDeconv(w::AbstractArray{T,N}, 
                    σ,
                    b,
                    lambda,
                    rho,
                    iters,
                    iso) where {T,N}

  return ADMMDeconv(σ, w, b, lambda, rho, iters, iso)
end


function ADMMDeconv(k::NTuple{N,Integer},
                    num_it::Integer,
                    σ = Flux.identity; 
                    iso::Bool = false,
                    init = Flux.glorot_uniform, 
                    groups = 1,
                    b = true) where {N}

  weight = Flux.convfilter(k, 1=>1; init=init, groups=groups)
  λ = abs.(Flux.glorot_uniform(1))
  ρ = abs.(Flux.glorot_uniform(1))
  bias_w = Flux.create_bias(weight, b, 1)
  ADMMDeconv(weight, σ, bias_w, λ, ρ, num_it, iso)
end


Flux.@functor ADMMDeconv weight, bias, λ, ρ


function (d::ADMMDeconv)(x::AbstractArray)
  d.λ = sqrt.(d.λ .^2 .+ 1f-6^2)
  d.ρ = sqrt.(d.ρ .^2 .+ 1f-6^2)
  
  res = tvd_fft(x, d.λ, d.ρ, d.weight, d.iso, d.iters)
  res = res .+ d.bias
  
  return σ.(res)
end
#------------------------------------------------------------------------------------------------------------------------------
