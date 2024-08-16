using Random

Random.seed!(42)

using CUDA, Flux, ProgressBars, Zygote, JLD2, Plots, DataFrames, CSV, MLUtils
Zygote.@nograd CUDA.ones
Zygote.@nograd CUDA.zeros
Zygote.@nograd CUDA.floor
Zygote.@nograd CUDA.ceil
Zygote.@nograd CUDA.repeat
Zygote.@nograd typeof

include("processing/datafeeder.jl")
include("utilities/cfg_parse.jl")
include("nets/net_build.jl")
include("metrics/psnr.jl")
include("metrics/gmsd.jl")
include("metrics/ssim.jl")
include("optim/reduce_rl_plateau.jl")


function get_datafeeders(cfg_json::Dict)
    target_shape = (cfg_json["im_shape"][1], cfg_json["im_shape"][2])
    train_feeder = ImageDataFeeder(cfg_json["train_data"]["x_path"], cfg_json["train_data"]["y_path"], ".png", target_shape, target_shape)
    eval_feeder = ImageDataFeeder(cfg_json["eval_data"]["x_path"], cfg_json["eval_data"]["y_path"], ".png", target_shape, target_shape)

    train_loader = Flux.DataLoader(train_feeder, batchsize=cfg_json["batch_size"], shuffle=true)
    eval_loader = Flux.DataLoader(eval_feeder, batchsize=cfg_json["batch_size"], shuffle=true)

    return train_loader, eval_loader
end


function save_model(model_fpath::AbstractString, model)
	let model = cpu(model)
		model_state = Flux.state(model)
		jldsave(model_fpath; model_state)
	end
end


function run_train(xy_train::Flux.DataLoader, modelref, opt, metrics::Vector{Function}, train_results_arr::Vector)
	println("\nTRAINING")
	
	step_results = zeros((length(train_results_arr),))

	loss_f = metrics[1]

	for (x,y) in ProgressBar(xy_train)
		
		out = modelref(x)

		# ∂L∂m = Flux.gradient(loss_f, modelref, observ...)[1]
		res_err, ∂L∂m = Flux.withgradient(modelref) do m
			loss_f(m, x, y)
		end
		Flux.update!(opt, modelref, ∂L∂m[1])

		step_results[1] = res_err
		step_res_msg = "train_loss= $res_err"
		for (i, metric) in enumerate(metrics[2:end])
			step_results[i+1] = metric(out, y)
			step_res_msg *= "; train_$(String(Symbol(metric))) = $(step_results[i+1])"
		end
	
		print(step_res_msg)

		train_results_arr += step_results

		GC.gc();
		CUDA.reclaim()
	end

	train_results_arr ./= length(xy_train)
	avg_res_msg = "\n\nepoch_train_loss = $(train_results_arr[1])"
	for (i, metric) in enumerate(metrics[2:end])
		avg_res_msg *= "; epoch_train_$(String(Symbol(metric))) = $(train_results_arr[i+1])"
	end

	printstyled(avg_res_msg, bold=true, color=:green)

	return train_results_arr
end


function run_eval(xy_eval::Flux.DataLoader, model, opt, metrics::Vector{Function}, eval_results_arr::Vector)
	println("\n\nVALIDATING")

	@assert length(eval_results_arr) == (length(metrics))

	loss_f = metrics[1]

	for (x,y) in ProgressBar(xy_eval)
		out = model(x)
		eval_results_arr[1] += loss_f(model, x, y)

		for (i, metric) in enumerate(metrics[2:end])
			eval_results_arr[i+1] += metric(out, y)
		end
	end

	eval_results_arr ./= length(xy_eval)
	eval_msg = "\nepoch_vloss = $(eval_results_arr[1])"
	for (i, metric) in enumerate(metrics[2:end])
		eval_msg *= "; epoch_val_$(String(Symbol(metric))) = $(eval_results_arr[i+1])"
	end

	printstyled(eval_msg, bold=true, color=:green)

	return eval_results_arr
end


function train_model(train_eval::Tuple, 
                     modelcfg::Dict, 
                     loss_func::Function,
					 model_name::AbstractString,
                     to_device=cpu)

	train_loader = train_eval[1]|>to_device
	eval_loader = train_eval[2]|>to_device

	# Create model and config params
    model = admm_denoiser(modelcfg)|>to_device
	optim = Flux.Optimiser(AdaBelief(modelcfg["lr_rate"])) 
	opt = Flux.setup(optim, model)

	rl_plateau_reducer = ReduceRLPlateau(optim, 4, 0.5)

	# Instantiate loss and metrics functions
    loss(m, x, y) = loss_func(m(x), y)|>to_device
	gmsd_m(x, y) = gmsd_loss(x, y)|>to_device
    psnr_m(x, y) = peak_snr(x, y)|>to_device
    mmse_m(x, y) = Flux.mse(x, y)|>to_device

	metrics_arr = [loss, gmsd_m, psnr_m, mmse_m]

	best_val_loss = Inf

	# Create directory path where trained models will be saved based on given model name
	save_model_dir = get_project_root() * "/trained_models/$model_name"
	save_train_history_path = save_model_dir * "/train_eval_metrics_history.csv"
	mkpath(save_model_dir)

	training_res_df = DataFrame()
	for stage in ["train", "eval"]
		for metric in metrics_arr
			col = "$(stage)_$(String(Symbol(metric)))"
			training_res_df[!, col] = []
		end
	end

    for epoch in 1:modelcfg["epochs"]
		println("\n\n\n\t\t\t\t\t\t\t\t\t\t\t\t\t[ EPOCH $epoch ]")

		epoch_train_res_arr = zeros((length(metrics_arr), ))
		epoch_eval_res_arr = zeros((length(metrics_arr), ))
		
		epoch_train_res_arr = run_train(train_loader, model, opt, metrics_arr, epoch_train_res_arr)
		epoch_eval_res_arr = run_eval(eval_loader, model, opt, metrics_arr, epoch_eval_res_arr)

		onplateau!(rl_plateau_reducer, epoch_eval_res_arr[1], model, opt)
		
		if epoch_eval_res_arr[1] < best_val_loss
			model_path = save_model_dir * "/$model_name-ep_$epoch-vloss_$(round(epoch_eval_res_arr[1], digits=4))-psnr_$(round(epoch_eval_res_arr[2], digits=4))-mse_$(round(epoch_eval_res_arr[3], digits=4)).jld2"
			save_model(model_path, model)
			best_val_loss = epoch_eval_res_arr[1]
		end

		epoch_res = cat(epoch_train_res_arr, epoch_eval_res_arr, dims=1)
		push!(training_res_df, epoch_res)
		CSV.write(save_train_history_path, training_res_df)

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
	train_model(trainf_evalf, train_cfg, ssim_loss, user_args["model_name"], Flux.gpu)
	printstyled("\nDONE", bold=true, color=:green)
end


main()
