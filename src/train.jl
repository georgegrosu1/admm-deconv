using ArgParse, Flux, ProgressBars, Zygote

include("processing/datafeeder.jl")
include("utilities/cfg_parse.jl")


function main()
    user_args = parse_terminal_args()

    train_cfg = fetch_json_data(user_args["cfg_fname"])

    println(train_cfg)
end

main()