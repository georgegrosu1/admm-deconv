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
    opt = nothing
    for attr in optimizer
      if typeof(attr) <: Flux.Optimise.AbstractOptimiser
        opt = attr
      end
    end
    
    return ReduceRLPlateau(opt, patience, init_obs, factor, tolerance, counter)
  end


  function onplateau!(red_rl::ReduceRLPlateau, loss_val::Number, model, opt_state)
    if abs(red_rl.last_obs - red_rl.last_obs * red_rl.tolerance) <= loss_val
        red_rl.counter += 1
    else
        red_rl.counter = 0
        red_rl.last_obs = loss_val
    end

    if red_rl.counter == red_rl.patience
        red_rl.optimizer.eta -= red_rl.optimizer.eta * red_rl.reduce_factor
        println("\nReducing LR to: $(red_rl.optimizer.eta)")
        red_rl.counter = 0
        red_rl.last_obs = loss_val

        opt_state = Flux.setup(red_rl.optimizer, model)
    end
  end