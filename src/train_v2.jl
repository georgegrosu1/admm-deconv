using Random

Random.seed!(42)

using FastAI, FluxTraining, Flux, Zygote, CUDA

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


function train_model(trainloader::DataLoader, 
					 testloader::DataLoader,
					 modelcfg::Dict, 
                     loss_func::Function,
					 metrics::Vector{Function},
					 model_name::AbstractString)


	# Make model directory
	model_dir = get_project_root() * "/trained_models/$model_name"
	mkpath(model_dir)

	# Create model and config params
    model = admm_denoiser(modelcfg)
	optim = Flux.AdaMax(modelcfg["lr_rate"], (0.9, 0.999), 1.0e-8)

	# Set metrics
	metrics = [
		FluxTraining.Metric(m, device=gpu) 
		for m in metrics
	]
	cbmetrics = FluxTraining.Metrics(metrics...)

	# Set logging
	log_dir = model_dir * "/logging"
	mkpath(log_dir)
	backend = TensorBoardBackend(log_dir)
	logmetrics = LogMetrics(backend)
	loghist = LogHistograms(backend)
	ckp = Checkpointer(model_dir)

	cbs = [ToGPU(), cbmetrics, logmetrics, loghist, ckp]

	learner = FluxTraining.Learner(model, loss_func; callbacks=cbs, optimizer=optim)

    FluxTraining.fit!(learner, modelcfg["epochs"], (trainloader, testloader))
end


function main()
	printstyled("\nParsing args", bold=true)
    user_args = parse_terminal_args()
	printstyled("\nDONE", bold=true, color=:green)

	printstyled("\nParsing train configs", bold=true)
    train_cfg = fetch_json_data(user_args["cfg_fname"])
	printstyled("\nDONE", bold=true, color=:green)

	printstyled("\nInitializing data feeders", bold=true)
    trainloader, testloader = get_datafeeders(train_cfg)
	printstyled("\nDONE", bold=true, color=:green)

	printstyled("\nProceeding with training", bold=true)
	train_model(trainloader, testloader, train_cfg, ssim_loss, [peak_snr, gmsd, Flux.mse], user_args["model_name"])
	printstyled("\nDONE", bold=true, color=:green)
end


main()
