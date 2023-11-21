"""
    convfilter(filter::Tuple, in => out[; init = glorot_uniform])

Constructs a standard convolutional weight matrix with given `filter` and
channels from `in` to `out`.

Accepts the keyword `init` (default: `glorot_uniform`) to control the sampling
distribution.

This is internally used by the [`Conv`](@ref) layer.
"""
function convfilter(filter::NTuple{N,Integer}, ch::Pair{<:Integer,<:Integer};
          init = glorot_uniform, groups = 1) where N
  cin, cout = ch
  @assert cin % groups == 0 "Input channel dimension must be divisible by groups."
  @assert cout % groups == 0 "Output channel dimension must be divisible by groups."
  init(filter..., cin÷groups, cout)
end

@functor Conv


struct DeconvADMM{N,M,F,A,V}
  σ::F
  weight::A
  bias::V
  stride::NTuple{N,Int}
  pad::NTuple{M,Int}
  dilation::NTuple{N,Int}
  groups::Int
end

"""
    Conv(weight::AbstractArray, [bias, activation; stride, pad, dilation])

Constructs a convolutional layer with the given weight and bias.
Accepts the same keywords and has the same defaults as
[`Conv(k::NTuple{N,Integer}, ch::Pair{<:Integer,<:Integer}, σ; ...)`](@ref Conv).

```jldoctest
julia> weight = rand(3, 4, 5);

julia> bias = zeros(5);

julia> layer = Conv(weight, bias, sigmoid)  # expects 1 spatial dimension
Conv((3,), 4 => 5, σ)  # 65 parameters

julia> layer(randn(100, 4, 64)) |> size
(98, 5, 64)

julia> Flux.params(layer) |> length
2
```
"""
function Conv(w::AbstractArray{T,N}, b = true, σ = identity;
              stride = 1, pad = 0, dilation = 1, groups = 1) where {T,N}

  @assert size(w, N) % groups == 0 "Output channel dimension must be divisible by groups."
  stride = expand(Val(N-2), stride)
  dilation = expand(Val(N-2), dilation)
  pad = calc_padding(Conv, pad, size(w)[1:N-2], dilation, stride)
  bias = create_bias(w, b, size(w, N))
  return Conv(σ, w, bias, stride, pad, dilation, groups)
end