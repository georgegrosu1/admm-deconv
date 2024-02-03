using Flux


mutable struct ReduceRLPlateau{O,A,B}
    optimizer::O
    patience::A
    last_obs::B
    reduce_factor::B
    tolerance::B

    counter::A
  end
  
  
  function ReduceRLPlateau(optimizer::Flux.Optimise.AbstractOptimiser, patience::Integer, factor::Real, tolerance::Real=0.03)
    counter = 0
    init_obs = prevfloat(typemax(Float64))
    return ReduceRLPlateau(optimizer, patience, init_obs, factor, tolerance, counter)
  end


  function (red::ReduceRLPlateau)(loss_val::Number)
    if abs(red.last_obs - red.last_obs * red.tolerance) <= loss_val
        red.counter += 1
    else
        red.counter = 0
        red.last_obs = loss_val
    end

    if red.counter == red.patience
        red.optimizer.eta -= red.optimizer.eta * red.reduce_factor
        println("Reducing LR to: $(red.optimizer.eta)")
        red.counter = 0
        red.last_obs = loss_val
    end
  end