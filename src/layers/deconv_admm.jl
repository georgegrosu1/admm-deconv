using Flux, Zygote, NNlib, DeconvOptim


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

  weight = Flux.convfilter(k, ch; init=init, groups=groups)
  λ = Flux.glorot_uniform(3)
  bias_w = Flux.create_bias(weight, b, 1)
  ADMMDeconv(weight, σ, bias_w, λ)
end


Flux.@functor ADMMDeconv weight, bias, λ


function (d::ADMMDeconv)(x::AbstractArray)

  σ = NNlib.fast_act(d.σ, x)
  # xT = Flux._match_eltype(d, x)

  # res, _ = DeconvOptim.deconvolution(xT, d.weight, regularizer=TV(num_dims=Flux.ndims(xT), sum_dims=[1, 2], weights=d.λ))
  res = x .* d.bias
  
  return σ.(res)
end
#------------------------------------------------------------------------------------------------------------------------------
