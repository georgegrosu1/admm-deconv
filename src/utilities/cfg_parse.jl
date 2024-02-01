using JSON, ArgParse


extension(file::AbstractString) = splitext(file)[end]

function fetch_json_data(js_file_name::AbstractString)
    if extension(js_file_name) != ".json"
        @error string("Config file has wrong file extension! .json is required but ", extension(js_file_name), " is given.")
    end
    config_file_path = get_configs_dir() * "/" * js_file_name
    return JSON.parsefile(config_file_path)
end


function get_project_root()
    return replace(dirname(Base.active_project()), "\\"=>"/")
end


function get_configs_dir()
    return get_project_root() * "/src/configs"
end


function parse_terminal_args()
    user_args = ArgParseSettings()

    @add_arg_table user_args begin
        "--cfg_fname", "-c"
            help = "String containing filename of training JSON config"
            arg_type = AbstractString
            default = "train_cfg.json"
        "--model_name", "-n"
            help = "Name of the model to be saved (name identifies with saved weights file name)"
            arg_type = AbstractString
            default = "admm-tv_restorer"
    end

    return parse_args(user_args)
end