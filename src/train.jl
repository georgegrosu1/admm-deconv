import Random
Random.seed!(90)

# ENV["JULIA_CUDA_MEMORY_POOL"] = "none"

using CUDA, Flux, ProgressBars, Zygote, JLD2, Plots, DataFrames, CSV
Zygote.@nograd CUDA.ones
Zygote.@nograd CUDA.zeros
Zygote.@nograd CUDA.floor
Zygote.@nograd CUDA.ceil
Zygote.@nograd CUDA.repeat
Zygote.@nograd typeof

include("processing/datafeeder.jl")
include("utilities/cfg_parse.jl")
include("nets/net_build.jl")
include("metrics/iqi.jl")
include("optim/reduce_rl_plateau.jl")


function get_datafeeders(cfg_json::Dict)
    target_shape = (cfg_json["im_shape"][1], cfg_json["im_shape"][2])
    train_feeder = ImageDataFeeder(cfg_json["train_data"]["x_path"], cfg_json["train_data"]["y_path"], ".png", target_shape, target_shape)
    eval_feeder = ImageDataFeeder(cfg_json["eval_data"]["x_path"], cfg_json["eval_data"]["y_path"], ".png", target_shape, target_shape)

    train_loader = Flux.DataLoader(train_feeder, batchsize=cfg_json["batch_size"])
    eval_loader = Flux.DataLoader(eval_feeder, batchsize=cfg_json["batch_size"])

    return train_loader, eval_loader
end


function save_model(model_fpath::AbstractString, model_state)
	jldsave(model_fpath; model_state)
end


function run_train(xy_train::Flux.DataLoader, modelref, opt, loss_f::Function, metrics::Vector{Function})
	println("\nTRAINING")

	avg_results = zeros((length(metrics) + 1,))
	step_results = zeros((length(metrics) + 1,))

	for (x,y) in ProgressBar(xy_train)
		out = modelref(x)
		res_err, grads = Flux.withgradient(modelref) do m
			loss_f(m(x), y)
		end
		Flux.update!(opt, modelref, grads[1])

		step_results[1] = res_err
		step_res_msg = "train_loss= $res_err"

		for (i, metric) in enumerate(metrics)
			step_results[i+1] = metric(out, y)
			step_res_msg *= "; train_$(String(Symbol(metric))) = $(step_results[i+1])"
		end
	
		print(step_res_msg)

		avg_results += step_results

		GC.gc();
		CUDA.reclaim()
	end

	avg_results ./= length(xy_train)
	avg_res_msg = "\n\nepoch_train_loss= $(avg_results[1])"
	for (i, metric) in enumerate(metrics)
		avg_res_msg *= "; epoch_train_$(String(Symbol(metric))) = $(step_results[i+1])"
	end

	printstyled(avg_res_msg, bold=true, color=:green)
end


function run_eval!(xy_eval::Flux.DataLoader, model, opt, loss_f::Function, metrics::Vector{Function}, avg_results::Vector)
	println("\n\nVALIDATING")

	@assert length(avg_results) == (length(metrics) + 1)

	for (x,y) in ProgressBar(xy_eval)
		out = model(x)
		avg_results[1] += loss_f(out, y)

		for (i, metric) in enumerate(metrics)
			avg_results[i+1] += metric(out, y)
		end
	end

	avg_results ./= length(xy_eval)
	eval_msg = "\nepoch_vloss= $(avg_results[1])"
	for (i, metric) in enumerate(metrics)
		eval_msg *= "; epoch_val_$(String(Symbol(metric))) = $(avg_results[i+1])"
	end

	printstyled(eval_msg, bold=true, color=:green)
end


function train_model(train_eval::Tuple, 
                     modelcfg::Dict, 
                     loss_func::Function,
					 model_name::AbstractString,
                     to_device=cpu)

	train_loader = train_eval[1]|>to_device
	eval_loader = train_eval[2]|>to_device

	# Create model and config params
    model = admm_restoration_model(modelcfg)|>to_device
	optim = Flux.Optimiser(AdaBelief(modelcfg["lr_rate"])) 
	opt = Flux.setup(optim, model)

	rl_plateau_reducer = ReduceRLPlateau(optim, 1, 0.5)

	# Instantiate loss and metrics functions
    loss(x, y) = loss_func(x, y)|>to_device
    psnr(x, y) = peak_snr(x, y)|>to_device
    mmse(x, y) = Flux.mse(x, y)|>to_device

	best_val_loss = Inf

	# Create directory path where trained models will be saved based on given model name
	save_model_dir = get_project_root() * "/trained_models/$model_name"
	mkpath(save_model_dir)

    for epoch in 1:modelcfg["epochs"]
		println("\n\n\n\t\t\t\t\t\t\t\t\t\t\t\t\t[ EPOCH $epoch ]")
		
		run_train(train_loader, model, opt, loss, [psnr, mmse])

		avg_eval_res = zeros(3)
		run_eval!(eval_loader, model, opt, loss, [psnr, mmse], avg_eval_res)

		onplateau!(rl_plateau_reducer, avg_eval_res[1], model, opt)
		
		if avg_eval_res[1] < best_val_loss
			model_path = save_model_dir * "/$model_name-ep_$epoch-vloss_$(round(avg_eval_res[1], digits=4))-psnr_$(round(avg_eval_res[2], digits=4))-mse_$(round(avg_eval_res[3], digits=4)).jld2"
			save_model(model_path, Flux.state(model))
			best_val_loss = avg_eval_res[1]
		end
		printstyled("
		\n-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
	end
end


function main()
	printstyled("\nParsing args", bold=true)
    user_args = parse_terminal_args()
	printstyled("\nDONE", bold=true, color=:green)

	printstyled("\nParsing train configs", bold=true)
    train_cfg = fetch_json_data(user_args["cfg_fname"])
	printstyled("\nDONE", bold=true, color=:green)

	printstyled("\nInitializing data feeders", bold=true)
    trainf_evalf = get_datafeeders(train_cfg)
	printstyled("\nDONE", bold=true, color=:green)

	printstyled("\nProceeding with training", bold=true)
	train_model(trainf_evalf, train_cfg, Flux.mae, user_args["model_name"], Flux.gpu)
	printstyled("\nDONE", bold=true, color=:green)
end


main()
