using Flux

include("../layers/deconv_admm.jl")


bcat(x...) = cat(x..., dims=4)

function admm_restoration_model(mcfg::Dict)
    # branch1 = Chain(
    #     ADMMDeconv((10,10), relu)
    #     )

    # branch2 = Chain(
    #     ADMMDeconv((10,10), relu)
    #     )

    model = Chain(
            ADMMDeconv((10,10), 30, relu6),
            Conv((1,1), 3=>3, relu6))

    return model
end