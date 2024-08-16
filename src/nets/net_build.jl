using Flux, CUDA

include("../layers/deconv_admm.jl")


chcat(x...) = cat(x..., dims=3)

relu1(x) = min.(relu(x), 1)


struct Activation{F}
    activfn::F
end

(act::Activation)(x::AbstractArray) = act.activfn.(x)

Normalization(x::AbstractArray) = Flux.normalise(x, dims=(1,2,3))


function updownblock(upker, downker, upfilters::Pair, downfilters::Pair)
    return Chain(
        ConvTranspose(upker, upfilters, stride=1, pad=0; init=Flux.orthogonal),
        Conv(downker, downfilters, stride=1, pad=0; init=Flux.orthogonal),
        Normalization,
        relu6
    )
end


function downblock(downker, downfilters::Pair, pool_win::Tuple)
    return Chain(
        Conv(downker, downfilters, stride=1, pad=0; init=Flux.orthogonal),
        Normalization,
        MaxPool(pool_win, pad=SamePad(), stride=1),
        relu6
    )
end


function upblock(upker, upfilters::Pair, pool_win::Tuple)
    return Chain(
        ConvTranspose(upker, upfilters, stride=1, pad=0; init=Flux.orthogonal),
        Normalization,
        MaxPool(pool_win, pad=SamePad(), stride=1),
        relu6
    )
end


function updownresidualblock(layers, upker, downker, uf, df)
    fwd = Chain(layers...)
    updown = Chain(updownblock(upker, downker, uf, df), updownblock(upker, downker, df, df))
    return Parallel(chcat, fwd, updown)
end


function get_autoencoder(mcfg::Dict)

    ker1, ker2, ker3, ker4, ker5, ker6 = (23, 23), (21, 21), (17, 17), (15, 15), (11, 11), (9, 9)
    afd1, afd2, afd3, afd4, afd5, afd6 = 3=>16, 16=>16, 16=>32, 32=>32, 32=>64, 64=>64

    upkresid = 32

    afu1 = last(afd6)=>16
    afu2 = (last(afu1)+upkresid)=>64
    afu3 = (last(afu2)+upkresid)=>64
    afu4 = (last(afu3)+upkresid)=>64
    afu5 = (last(afu4)+upkresid)=>64
    afu6 = (last(afu5)+upkresid)=>128
    
    down_1 = downblock(ker1, afd1, (3,3))
    down_2 = downblock(ker2, afd2, (3,3))
    down_3 = downblock(ker3, afd3, (3,3))
    down_4 = downblock(ker4, afd4, (5,5))
    down_5 = downblock(ker5, afd5, (5,5))
    down_6 = downblock(ker6, afd6, (7,7))

    up_1 = upblock(ker6, afu1, (3,3))
    up_2 = upblock(ker5, afu2, (3,3))
    up_3 = upblock(ker4, afu3, (3,3))
    up_4 = upblock(ker3, afu4, (5,5))
    up_5 = upblock(ker2, afu5, (7,7))
    up_6 = upblock(ker1, afu6, (3,3))

    ud_res1 = updownresidualblock([down_6, up_1], (3,3), (3,3), last(afd5)=>upkresid, upkresid=>upkresid)
    ud_res2 = updownresidualblock([down_5, ud_res1, up_2], (5,5), (5,5), last(afd4)=>upkresid, upkresid=>upkresid)
    ud_res3 = updownresidualblock([down_4, ud_res2, up_3], (9,9), (9,9), last(afd3)=>upkresid, upkresid=>upkresid)
    ud_res4 = updownresidualblock([down_3, ud_res3, up_4], (7,7), (7,7), last(afd2)=>upkresid, upkresid=>upkresid)
    ud_res5 = updownresidualblock([down_2, ud_res4, up_5], (5,5), (5,5), last(afd1)=>upkresid, upkresid=>upkresid)
    autoencoder = updownresidualblock([down_1, ud_res5, up_6], (3,3), (3,3), 3=>upkresid, upkresid=>upkresid)
    
    # skip1 = SkipConnection(Chain(down_5, up_1), chcat)
    # skip2 = SkipConnection(Chain(down_4, skip1, up_2), chcat)
    # skip3 = SkipConnection(Chain(down_3, skip2, up_3), chcat)
    # skip4 = SkipConnection(Chain(down_2, skip3, up_4), chcat)
    # autoencoder = Chain(down_1, skip4, up_5)

    return autoencoder
end


function get_deconvs()

    k1, k2, k3 = (7, 7), (10, 10), (15, 15)
    λ1, λ2, λ3 = 0.004f0, 0.04f0, 0.4f0
    ρ1, ρ2, ρ3 = 0.02f0, 0.04f0, 0.06f0
    deconv_1 = ADMMDeconvF3(k1, 50, λ1, ρ1, relu6, iso=mcfg["use_iso"])
    deconv_2 = ADMMDeconvF3(k2, 50, λ2, ρ2, relu6, iso=mcfg["use_iso"])
    deconv_3 = ADMMDeconvF3(k3, 50, λ3, ρ3, relu6, iso=!mcfg["use_iso"])
end


function get_denoiser(mcfg::Dict)
    
    deconv_1 = ADMMDeconvF2((), 50, 0.002f0, relu1, iso=mcfg["use_iso"])
    deconv_2 = ADMMDeconvF2((), 50, 0.02f0, relu1, iso=mcfg["use_iso"])
    deconv_3 = ADMMDeconvF2((), 50, 0.2f0, relu1, iso=mcfg["use_iso"])
    deconv_4 = ADMMDeconvF2((), 50, 2f0, relu1, iso=mcfg["use_iso"])
    deconv_5 = ADMMDeconvF2((), 50, 4f0, relu1, iso=mcfg["use_iso"])

    deconvs = Parallel(
        chcat, 
        deconv_1, deconv_2,
        deconv_3, deconv_4, deconv_5
        )

    return deconvs
end


function get_multistage_updownscale(mcfg::Dict)

    adk = (10, 10)
    uk1, uk2, uk3, uk4 = (25, 25), (19, 19), (13, 13), (9, 9)
    dk1, dk2, dk3, dk4 = (9, 9), (7, 7), (5, 5), (3, 3)
    fu1 = 38=>32
    fd1 = last(fu1)=>32

    fu2 = last(fd1)=>32
    fd2 = last(fu2)=>64

    fu3 = last(fd2)=>64
    fd3 = last(fu3)=>64

    fu4 = last(fd3)=>64
    fd4 = last(fu4)=>64

    fu5 = (last(fd4))=>32
    fd5 = last(fu5)=>32

    fu6 = (last(fd5)+last(fd1))=>32
    fd6 = last(fu6)=>32
    norm1, norm2 = BatchNorm, InstanceNorm

    admm1 = ADMMDeconv(adk, 50, relu, iso=mcfg["use_iso"])
    ud_1 = updownblock(uk1, dk1, fu1, fd1)
    ud_2 = updownblock(uk2, dk2, fu2, fd2)
    ud_3 = updownblock(uk3, dk3, fu3, fd3)
    ud_4 = updownblock(uk4, dk4, fu4, fd4)
    ud_5 = updownblock(uk4, dk4, fu5, fd5)
    ud_6 = updownblock(uk4, dk4, fu6, fd6)

    skip_ud_34 = SkipConnection(Chain(ud_3, ud_4), +)
    skip_ud_2345 = SkipConnection(Chain(ud_2, skip_ud_34, ud_5), chcat)
    updownscale = Chain(admm1, ud_1, skip_ud_2345, ud_6)
end


function admm_denoiser(mcfg::Dict)
    
    # input = Chain(identity)
    autoencoder = get_autoencoder(mcfg)
    denoiser = get_denoiser(mcfg)

    auto_denoise = Parallel(chcat, autoencoder, denoiser)

    last_updown = updownblock((5,5), (5, 5), 175=>32, 32=>32)
    last_updown2 = updownblock((5,5), (5, 5), 35=>32, 32=>3)

    core = Chain(auto_denoise, last_updown)
    prefin = SkipConnection(core, chcat)
    fin = Chain(prefin, last_updown2, relu1)

    paramCount = 0
    for layer in fin
        paramCount += sum(length, Flux.params(layer), init=0)
    end

    printstyled("\n\nMODEL SIZE (#parameters): $paramCount", bold=true, italic=true, color=:light_magenta)

    return fin
end
