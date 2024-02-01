using Flux


mutable struct ReduceRLPlateau{O,A,B,P}
    optimizer::O
    patience::A
    last_obs::B
    reduce_factor::B
    min_tolerance::B
    dist_f::P

    counter::A
    initial_lr::B
  end
  
  
  function ReduceRLPlateau(optimizer::O, patience::Integer, factor::Real, min_tolerance::Real=1e-4, dist::P=-) where {O,P}
    counter = 0
    initial_lr = optimizer.eta
    init_obs = prevfloat(typemax(Float64))
    return ReduceRLPlateau(optimizer, patience, init_obs, factor, min_tolerance, dist, counter, initial_lr)
  end


  function (red::ReduceRLPlateau)(loss_val::Number)
    if abs(red.dist_f(red.last_obs, loss_val)) <= red.min_tolerance
        red.counter += 1
    else
        red.counter = 0
    end

    if red.counter == red.patience
        red.optimizer.eta -= red.initial_lr * red.factor
        println("Reducing LR to: $(red.optimizer.eta)")
        red.counter = 0
    end

    red.last_obs = loss_val
  end