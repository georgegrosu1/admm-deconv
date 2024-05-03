using Flux

include("../layers/deconv_admm.jl")


chcat(x...) = cat(x..., dims=3)

function admm_restoration_model(mcfg::Dict)

    k_admm1, k_admm2, k_admm3 = 10, 15, 20
    k_convt1, k_convt2, k_convt3 = 38, 20, 16
    k_conv1, k_conv2, k_conv3 = div(k_convt1, 2), div(k_convt2, 2), div(k_convt3, 2)
    f_conv1, f_conv2, f_conv3 = 18, 32, 64

    updownscale_branch = Chain(
        ADMMDeconv((k_admm, k_admm), 50, relu6, iso=mcfg["use_iso"]),
        ConvTranspose((k_convt1,k_convt1), 3=>f_conv1),
        Conv((k_conv1, k_conv1), f_conv1=>f_conv1),
        AdaptiveMaxPool(mcfg["im_shape"])
        BatchNorm(f_conv1, relu6),

        ConvTranspose((k_convt2,k_convt2), f_conv1=>f_conv2),
        Conv((k_conv2, k_conv2), f_conv2=>f_conv2),
        AdaptiveMaxPool(mcfg["im_shape"])
        BatchNorm(f_conv2, relu6),

        ConvTranspose((k_convt3,k_convt3), f_conv2=>f_conv3),
        Conv((k_conv3, k_conv3), f_conv3=>f_conv3),
        AdaptiveMaxPool(mcfg["im_shape"])
        BatchNorm(f_conv3, relu6),
    )

    autoencoder_branch = Chain(
        ADMMDeconv((k_admm2, k_admm2), 50, relu6, iso=mcfg["use_iso"]),
        ConvTranspose((k_convt1,k_convt1), 3=>3),
        BatchNorm(3, relu6),
        
        Conv((k_conv1, k_conv1), f_conv1=>f_conv1),
        BatchNorm(f_conv1, relu6),

        Conv((k_conv2, k_conv2), f_conv1=>f_conv1),
        BatchNorm(f_conv1, relu6),


        Conv((k_conv3, k_conv3), f_conv1=>f_conv1),
        BatchNorm(f_conv1, relu6),

        AdaptiveMaxPool(mcfg["im_shape"]),

        ConvTranspose((k_convt3,k_convt3), f_conv1=>f_conv1),
        BatchNorm(f_conv1, relu6),

        ConvTranspose((k_convt2,k_convt2), f_conv1=>f_conv2),
        BatchNorm(f_conv2, relu6),

        ConvTranspose((k_convt3,k_convt3), f_conv2=>f_conv3),
        BatchNorm(f_conv3, relu6),

        AdaptiveMaxPool(mcfg["im_shape"])
    )
        
    branches = Parallel(
        chcat,
        updownscale_branch,
        autoencoder_branch
    )

    fin = Chain(
        branches,
        ConvTranspose((9,9), 2*f_conv3=>3),
        BatchNorm(3, relu6),
        AdaptiveMaxPool(mcfg["im_shape"])
    )

    paramCount = 0
    for layer in fin
        paramCount += sum(length, Flux.params(layer))
    end

    printstyled("\n\nMODEL SIZE (#parameters): $paramCount", bold=true, italic=true, color=:light_magenta)

    return fin
end
