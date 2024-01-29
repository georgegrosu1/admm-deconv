using Flux

include("../layers/deconv_admm.jl")


bcat(x...) = cat(x..., dims=4)

function admm_restoration_model(mcfg::Dict)
    branch1 = Chain(
        ADMMDeconv((10,10), relu)
        )

    branch2 = Chain(
        ADMMDeconv((10,10), relu)
        )
    
end