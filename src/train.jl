import Random
Random.seed!(256)

using CUDA, Flux, ProgressBars, Zygote, JLD2
Zygote.@nograd CUDA.ones
Zygote.@nograd CUDA.zeros

include("processing/datafeeder.jl")
include("utilities/cfg_parse.jl")
include("nets/net_build.jl")
include("metrics/iqi.jl")
include("optim/reduce_rl_plateau.jl")


function get_datafeeders(cfg_json::Dict)
    x_shape = (cfg_json["x_shape"][1], cfg_json["x_shape"][2])
    y_shape = (cfg_json["y_shape"][1], cfg_json["y_shape"][2])
    train_feeder = ImageDataFeeder(cfg_json["train_data"]["x_path"], cfg_json["train_data"]["y_path"], ".png", x_shape, y_shape)
    eval_feeder = ImageDataFeeder(cfg_json["eval_data"]["x_path"], cfg_json["eval_data"]["y_path"], ".png", x_shape, y_shape)

    train_loader = Flux.DataLoader(train_feeder, batchsize=cfg_json["batch_size"])
    eval_loader = Flux.DataLoader(eval_feeder, batchsize=cfg_json["batch_size"])

    return train_loader, eval_loader
end


function save_model(model_fpath::AbstractString, model_state)
	jldsave(model_fpath; model_state)
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
    ps = Flux.params(model)
    opt = Flux.ADAM(modelcfg["lr_rate"])

	rl_plateau_reducer = ReduceRLPlateau(opt, 1, 0.5)

	# Instantiate loss and metrics functions
    loss(x, y) = loss_func(x, y)|>to_device
    psnr(x, y) = peak_snr(x, y)|>to_device
    mmse(x, y) = Flux.mse(x, y)|>to_device

	# Instantiate variables to store training and evaluation progress
    train_loss, train_mse, train_psnr = [], [], []
    val_loss, val_mse, val_psnr = [], [], []
	best_val_loss = Inf

	# Create directory path where trained models will be saved based on given model name
	save_model_dir = get_project_root() * "/trained_models/$model_name"
	mkpath(save_model_dir)

    for epoch in 1:modelcfg["epochs"]
		println("\n\n\n\t\t\t\t\t\t\t\t\t\t\t\t\t[ EPOCH $epoch ]")

		println("\nTRAINING")
		avg_train_err, avg_train_psnr, avg_train_mse = 0, 0, 0
        for (x,y) in ProgressBar(train_loader)
			out = model(x)
			gs = Flux.gradient(() -> loss(out, y), ps)
			Flux.update!(opt, ps, gs)

			res_err, res_psnr, res_mse = loss(out, y), psnr(out, y), mmse(out, y)
      
			print("train_loss= $res_err; train_psnr= $res_psnr; train_mse= $res_mse")

			avg_train_err += res_err
			avg_train_psnr += res_psnr
			avg_train_mse += res_mse
        end
		avg_train_err /= length(train_eval[1])
		avg_train_psnr /= length(train_eval[1])
		avg_train_mse /= length(train_eval[1])
		printstyled("\n\nepoch_train_loss= $avg_train_err; epoch_train_psnr= $avg_train_psnr; epoch_train_mse= $avg_train_mse", bold=true, color=:green)

		println("\n\nVALIDATING")
		avg_val_err, avg_val_psnr, avg_val_mse = 0, 0, 0
		for (x,y) in ProgressBar(eval_loader)
			out = model(x)
			avg_val_err += loss(out, y)
			avg_val_psnr += psnr(out, y)
			avg_val_mse += mmse(out, y)
		end

		rl_plateau_reducer(avg_val_err)

		avg_val_err /= length(train_eval[2])
		avg_val_psnr /= length(train_eval[2])
		avg_val_mse /= length(train_eval[2])
		printstyled("\nepoch_val_loss= $avg_val_err; epoch_val_psnr= $avg_val_psnr; epoch_val_mse= $avg_val_mse", bold=true, color=:green)
		printstyled("\n--------------------------------------------------------------------------------------------------------------------", color=:purple)

		if avg_val_err < best_val_loss
			model_path = save_model_dir * "/$model_name-ep_$epoch-vloss_$avg_val_err-psnr_$avg_val_psnr-mse_$avg_val_mse.jld2"
			save_model(model_path, Flux.state(model))
			best_val_loss = avg_val_err
		end
	end
end


function main()
    user_args = parse_terminal_args()

    train_cfg = fetch_json_data(user_args["cfg_fname"])

    trainf_evalf = get_datafeeders(train_cfg)

	train_model(trainf_evalf, train_cfg, ssim, user_args["model_name"], Flux.gpu)
end


main()
