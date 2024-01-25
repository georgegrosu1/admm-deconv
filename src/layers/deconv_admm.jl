using Flux, CUDA, NNlibCUDA
include("../ops/ops.jl")


#------------------------------------------------------------------------------------------------------------------------------
struct ADMMDeconv{F,A,N,V,M}
  σ::F
  weight::A
  bias::V
  λ::N
  ρ::M
end


function ADMMDeconv(w::AbstractArray{T,N}, 
                    σ,
                    b,
                    lambda,
                    rho) where {T,N}

  return ADMMDeconv(σ, w, b, lambda, rho)
end


function ADMMDeconv(k::NTuple{N,Integer},
                    σ = Flux.identity; 
                    init = Flux.glorot_uniform, 
                    groups = 1,
                    b = true) where {N}

  weight = Flux.convfilter(k, 1=>1; init=init, groups=groups)
  λ = abs.(Flux.glorot_uniform(1))
  ρ = abs.(Flux.glorot_uniform(1))
  bias_w = Flux.create_bias(weight, b, 1)
  ADMMDeconv(weight, σ, bias_w, λ, ρ)
end


Flux.@functor ADMMDeconv weight, bias, λ, ρ


function (d::ADMMDeconv)(x::AbstractArray)
  res = tvd_fft(x, d.λ, d.ρ, d.weight)
  res = res .+ d.bias
  
  return σ.(res)
end
#------------------------------------------------------------------------------------------------------------------------------
