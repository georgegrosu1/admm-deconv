using Flux, CUDA
include("../ops/ops.jl")


#------------------------------------------------------------------------------------------------------------------------------
mutable struct ADMMDeconvF1{F,A,N,V,M,B,C,D}
  σ::F
  weight::A
  bias::V
  λ::N
  ρ::M
  iters::B
  iso::C
  creg::D
end


function ADMMDeconvF1(w::AbstractArray{T,N}, 
                    σ,
                    b,
                    lambda,
                    rho,
                    iters,
                    iso,
                    creg) where {T,N}

  return ADMMDeconvF1(σ, w, b, lambda, rho, iters, iso, creg)
end


function ADMMDeconvF1(k::NTuple{N,Integer},
                      num_it::Integer,
                      λ,
                      σ = Flux.identity; 
                      iso::Bool = false,
                      init = Flux.glorot_uniform, 
                      groups = 1,
                      bias = false,
                      creg::Number = 0f0) where {N}


@assert λ > 0f0 "Parameter λ must be greater than 0"

weight = Flux.convfilter(k, 1=>1; init=init, groups=groups)
λ = zeros(1) .+ λ
ρ = abs.(Flux.glorot_uniform(1))
bias_w = Flux.create_bias(weight, bias, 1)
ADMMDeconvF1(weight, σ, bias_w, λ, ρ, num_it, iso, creg)
end

Flux.@layer ADMMDeconvF1 trainable=(weight, bias, ρ,)


mutable struct ADMMDeconvF2{F,A,N,V,M,B,C,D}
  σ::F
  weight::A
  bias::V
  λ::N
  ρ::M
  iters::B
  iso::C
  creg::D
end


function ADMMDeconvF2(w::AbstractArray{T,N}, 
                    σ,
                    b,
                    lambda,
                    rho,
                    iters,
                    iso,
                    creg) where {T,N}

  return ADMMDeconvF2(σ, w, b, lambda, rho, iters, iso, creg)
end


function ADMMDeconvF2(k::NTuple{N,Integer},
                      num_it::Integer,
                      ρ,
                      σ = Flux.identity; 
                      iso::Bool = false,
                      init = Flux.glorot_uniform, 
                      groups = 1,
                      bias = false,
                      creg::Number = 0f0) where {N}


@assert ρ > 0 "Parameter ρ must be greater than 0"

weight = Flux.convfilter(k, 1=>1; init=init, groups=groups)
λ = abs.(Flux.glorot_uniform(1))
ρ = zeros(1) .+ ρ
bias_w = Flux.create_bias(weight, bias, 1)
ADMMDeconvF2(weight, σ, bias_w, λ, ρ, num_it, iso, creg)
end

Flux.@layer ADMMDeconvF2 trainable=(weight, bias, λ,)


mutable struct ADMMDeconvF3{F,A,N,V,M,B,C,D}
  σ::F
  weight::A
  bias::V
  λ::N
  ρ::M
  iters::B
  iso::C
  creg::D
end


function ADMMDeconvF3(w::AbstractArray{T,N}, 
                    σ,
                    b,
                    lambda,
                    rho,
                    iters,
                    iso,
                    creg) where {T,N}

  return ADMMDeconvF3(σ, w, b, lambda, rho, iters, iso, creg)
end


function ADMMDeconvF3(k::NTuple{N,Integer},
                      num_it::Integer,
                      λ,
                      ρ,
                      σ = Flux.identity; 
                      iso::Bool = false,
                      init = Flux.glorot_uniform, 
                      groups = 1,
                      bias = false,
                      creg::Number = 0f0) where {N}


@assert λ > 0 "Parameter λ must be greater than 0"
@assert ρ > 0 "Parameter ρ must be greater than 0"

weight = Flux.convfilter(k, 1=>1; init=init, groups=groups)
λ = zeros(1) .+ λ
ρ = zeros(1) .+ ρ
bias_w = Flux.create_bias(weight, bias, 1)
ADMMDeconvF3(weight, σ, bias_w, λ, ρ, num_it, iso, creg)
end

Flux.@layer ADMMDeconvF3 trainable=(weight, bias,)


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

Flux.@layer ADMMDeconv trainable=(weight, bias, λ, ρ,)


Admm = Union{ADMMDeconv, ADMMDeconvF1, ADMMDeconvF2, ADMMDeconvF3}


function (d::Admm)(x::AbstractArray)
  d.λ = clamp.(d.λ, d.creg, Inf32)
  d.ρ = clamp.(d.ρ, d.creg, Inf32)

  # d.weight = clamp.(d.weight, 0f0, Inf32)
  
  res = tvd_fft(x, d.λ, d.ρ, d.weight, d.iso, d.iters)
  res = res .+ d.bias
  
  return d.σ.(res)
end
#------------------------------------------------------------------------------------------------------------------------------
