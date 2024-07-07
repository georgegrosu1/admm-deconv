using Images, ProgressBars, Glob
include("utilities/base_funcs.jl")


DSET_DIR = replace(realpath(dirname(@__FILE__)*"/.."), "\\"=>"/") * "/datasets/RealBlur/"

function get_train_im_paths(train_txt_p::String)
    txt_lines = readlines(open(train_txt_p))

    x_y_paths = [(DSET_DIR .* split(xy, " ")[1:2]) for xy in txt_lines]

    return x_y_paths
end

function add_awgn(img::Matrix{RGB{N0f8}}, min_σ::Number=0, max_σ::Number=50, im_max_val::Number=255)
    im_arr = img2tensor(img) .* im_max_val

    std = rand(min_σ:max_σ)

    im_arr += std * randn(size(im_arr))
    clamp!(im_arr, 0, im_max_val)

    return tensor2img(im_arr ./ im_max_val)
end


function generate_dset(train_ims, min_σ, max_σ, dest)
    printstyled("Making dataset for $dest", bold=true, color=:green)

    x_dest = dest * "/x"
    y_dest = dest * "/y"

    mkpath(x_dest)
    mkpath(y_dest)

    for (idx, (y,x)) in ProgressBar(enumerate(train_ims))
        x_ext = fextension(x)
        y_ext = fextension(x)

        x_dest_file = x_dest * "/$idx.$x_ext"
        y_dest_file = y_dest * "/$idx.$y_ext"

        imx = Images.load(x)

        new_imx = add_awgn(imx, min_σ, max_σ)

        Images.save(x_dest_file, new_imx)
        cp(y, y_dest_file)
    end
end


function add_gopro(min_σ, max_σ)
    x_train_source = "D:/Projects/ISETC2022/dcnn-deblur/dataset/GOPRO_Large/train/origblur/x_set"
    y_train_source = "D:/Projects/ISETC2022/dcnn-deblur/dataset/GOPRO_Large/train/y_set"
    x_test_source = "D:/Projects/ISETC2022/dcnn-deblur/dataset/GOPRO_Large/test/origblur/x_set"
    y_test_source = "D:/Projects/ISETC2022/dcnn-deblur/dataset/GOPRO_Large/test/y_set"

    x_train_paths = Glob.glob("*.png", x_train_source)
    y_train_paths = Glob.glob("*.png", y_train_source)
    x_train_dest = DSET_DIR * "awgn_$(min_σ)_$(max_σ)/train/x/"
    y_train_dest = DSET_DIR * "awgn_$(min_σ)_$(max_σ)/train/y/"
    

    for (idx, (x, y)) in ProgressBar(enumerate(zip(x_train_paths, y_train_paths)))
        x_file_dest = x_train_dest * fname(x) * "_$idx.$(fextension(x))"
        y_file_dest = y_train_dest * fname(y) * "_$idx.$(fextension(y))"

        imx = Images.load(x)

        new_imx = add_awgn(imx, min_σ, max_σ)

        Images.save(x_file_dest, new_imx)
        cp(y, y_file_dest)
    end

    x_test_paths = Glob.glob("*.png", x_test_source)
    y_test_paths = Glob.glob("*.png", y_test_source)
    x_test_dest = DSET_DIR * "awgn_$(min_σ)_$(max_σ)/test/x/"
    y_test_dest = DSET_DIR * "awgn_$(min_σ)_$(max_σ)/test/y/"

    for (idx, (x, y)) in ProgressBar(enumerate(zip(x_test_paths, y_test_paths)))
        x_file_dest = x_test_dest * fname(x) * "_$idx.$(fextension(x))"
        y_file_dest = y_test_dest * fname(y) * "_$idx.$(fextension(y))"

        imx = Images.load(x)

        new_imx = add_awgn(imx, min_σ, max_σ)

        Images.save(x_file_dest, new_imx)
        cp(y, y_file_dest)
    end
end


function main()
    # train_txt_path_j = "D:/Projects/admm-deconv/datasets/RealBlur/RealBlur_J_train.txt"
    # train_txt_path_r = "D:/Projects/admm-deconv/datasets/RealBlur/RealBlur_R_train.txt"
    # test_txt_path_j = "D:/Projects/admm-deconv/datasets/RealBlur/RealBlur_J_test.txt"
    # test_txt_path_r = "D:/Projects/admm-deconv/datasets/RealBlur/RealBlur_R_test.txt"

    min_σ, max_σ = 0, 50

    # train_dest = DSET_DIR * "awgn_$(min_σ)_$(max_σ)/train"
    # test_dest = DSET_DIR * "awgn_$(min_σ)_$(max_σ)/test"

    # train_xy_paths = cat(get_train_im_paths(train_txt_path_j), get_train_im_paths(train_txt_path_r), dims=1)
    # test_xy_paths = cat(get_train_im_paths(test_txt_path_j), get_train_im_paths(test_txt_path_r), dims=1)

    # generate_dset(train_xy_paths, min_σ, max_σ, train_dest)
    # generate_dset(test_xy_paths, min_σ, max_σ, test_dest)

    add_gopro(min_σ, max_σ)
end


main()
