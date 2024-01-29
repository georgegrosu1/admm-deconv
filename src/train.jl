import Random
Random.seed!(256)

using Flux, ProgressBars, Zygote

include("processing/datafeeder.jl")
include("utilities/cfg_parse.jl")


function get_datafeeders(cfg_json::Dict)
    x_shape = (cfg_json["x_shape"][1], cfg_json["x_shape"][2])
    y_shape = (cfg_json["y_shape"][1], cfg_json["y_shape"][2])
    train_feeder = ImageDataFeeder(cfg_json["train_data"]["x_path"], cfg_json["train_data"]["y_path"], ".png", x_shape, y_shape)
    eval_feeder = ImageDataFeeder(cfg_json["eval_data"]["x_path"], cfg_json["eval_data"]["y_path"], ".png", x_shape, y_shape)

    train_loader = Flux.DataLoader(train_feeder, batchsize=cfg_json["batch_size"])
    eval_loader = Flux.DataLoader(eval_feeder, batchsize=cfg_json["batch_size"])

    return train_loader, eval_loader
end


function main()
    user_args = parse_terminal_args()

    train_cfg = fetch_json_data(user_args["cfg_fname"])

    trainf, evalf = get_datafeeders(train_cfg)
end

main()