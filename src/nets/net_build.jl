using Flux

include("../layers/deconv_admm.jl")


chcat(x...) = cat(x..., dims=3)

function admm_restoration_model(mcfg::Dict)

    k_admm = 15
    k_convt1 = 8
    k_convt2 = 16
    k_convt3 = 32

    updownscale_branch = Chain(
        ADMMDeconv((k_admm, k_admm), 50, relu6, iso=mcfg["use_iso"]),
    )

    autoencoder_branch = Chain(
        ADMMDeconv((k_admm, k_admm), 50, iso=mcfg["use_iso"]),
        ConvTranspose((k_convt1,k_convt1), 3=>16),
        BatchNorm(16, relu6),
        ConvTranspose((k_convt2,k_convt2), 16=>32),
        BatchNorm(32, relu6),
        ConvTranspose((k_convt3,k_convt3), 32=>64),
        BatchNorm(64, relu6),
        Conv((k_convt2,k_convt2), 64=>32),
        BatchNorm(32, relu6),
        Conv((k_convt3,k_convt3), 32=>16),
        BatchNorm(16, relu6),
        Conv((k_convt1,k_convt1), 16=>3),
        BatchNorm(3, relu6)
    )
        
    branches = Parallel(
        chcat,
        updownscale_branch,
        autoencoder_branch
    )

    fin = Chain(
        branches,
        ConvTranspose((9,9), 6=>18, relu6),
        BatchNorm(18, relu6),
        Conv((9,9), 18=>3, relu6),
        BatchNorm(3, relu6)
    )

    paramCount = 0
    for layer in fin
        paramCount += sum(length, Flux.params(layer))
    end

    printstyled("\n\nMODEL SIZE (#parameters): $paramCount", bold=true, italic=true, color=:light_magenta)

    return fin
end
