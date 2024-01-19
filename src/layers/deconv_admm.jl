using Flux, CUDA, NNlibCUDA
include("../ops/ops.jl")


#------------------------------------------------------------------------------------------------------------------------------
struct ADMMDeconv{F,A,N,V}
  σ::F
  weight::A
  bias::V
  λ::N
end


function ADMMDeconv(w::AbstractArray{T,N}, 
                    σ,
                    b,
                    lambda) where {T,N}

  return ADMMDeconv(σ, w, b, lambda)
end


function ADMMDeconv(k::NTuple{N,Integer}, 
                    ch::Pair{<:Integer,<:Integer}, 
                    σ = Flux.identity; 
                    init = Flux.glorot_uniform, 
                    groups = 1,
                    b = true) where N

  weight = Flux.convfilter(k, 1=>1; init=init, groups=groups)
  λ = Flux.glorot_uniform(1)
  bias_w = Flux.create_bias(weight, b, 1)
  ADMMDeconv(weight, σ, bias_w, λ)
end


Flux.@functor ADMMDeconv weight, bias, λ


function (d::ADMMDeconv)(x::AbstractArray)

  # σ = NNlibCUDA.fast_act(d.σ, x)
  # xT = Flux._match_eltype(d, x)

  res = tvd_fft_gpu(x, d.λ, CuArray{Float32}([1]), d.weight)
  res = res .* d.bias
  
  return σ.(res)
end
#------------------------------------------------------------------------------------------------------------------------------
