using Flux

include("../layers/deconv_admm.jl")


chcat(x...) = cat(x..., dims=3)


function updownblock(upker, downker, upfilters::Pair, downfilters::Pair, outshape)
    return Chain(
        ConvTranspose(upker, upfilters),
        BatchNorm(last(upfilters), relu),
        Conv(downker, downfilters),
        InstanceNorm(last(downfilters), relu, affine=true),
        AdaptiveMaxPool(outshape)
    )
end


function downblock(downker, downfilters::Pair, normtype, outshape)
    return Chain(
        Conv(downker, downfilters),
        normtype(last(downfilters), relu, affine=true),
        AdaptiveMaxPool(outshape)
    )
end


function upblock(upker, upfilters::Pair, normtype, outshape)
    return Chain(
        ConvTranspose(upker, upfilters),
        normtype(last(upfilters), relu, affine=true),
        AdaptiveMaxPool(outshape)
    )
end


function admm_restoration_model(mcfg::Dict)
    
    imshape = (mcfg["im_shape"][1], mcfg["im_shape"][2])


     # ↓ -------------------------------AUTOENCODER BRANCH ------------------------------- ↓ #

     downker1, downker2, downker3, downker4 = (3, 3), (7, 7), (13, 13), (19, 19)
     upker1, upker2, upker3, upker4 = (21, 21), (21, 21), (13, 13), (13, 13)
     afd1, afd2, afd3, afd4 = 3=>16, 16=>32, 32=>64, 64=>64
     afu1, afu2, afu3, afu4 = 64=>64, 128=>32, 64=>32, 48=>32
     norm1, norm2 = BatchNorm, InstanceNorm
     outs1, outs2, outs3, outs4 = (imshape[1]-10, imshape[1]-10), (imshape[1]-20, imshape[1]-20), (imshape[1]-40, imshape[1]-40), (imshape[1]-60, imshape[1]-60)
     down_1 = downblock(downker1, afd1, norm1, outs1)
     down_2 = downblock(downker2, afd2, norm2, outs2)
     down_3 = downblock(downker3, afd3, norm2, outs3)
     down_4 = downblock(downker4, afd4, norm2, outs4)
 
     up_1 = upblock(upker1, afu1, norm2, outs3)
     up_2 = upblock(upker2, afu2, norm2, outs2)
     up_3 = upblock(upker3, afu3, norm2, outs1)
     up_4 = upblock(upker4, afu4, norm2, imshape)
 
     skip1 = SkipConnection(Chain(down_4, up_1), chcat)
     skip2 = SkipConnection(Chain(down_3, skip1, up_2), chcat)
     skip3 = SkipConnection(Chain(down_2, skip2, up_3), chcat)
     autoencoder = SkipConnection(Chain(down_1, skip3, up_4), chcat)
 
     # ↑ -------------------------------AUTOENCODER BRANCH ------------------------------- ↑ #


    # ↓ -------------------------------UPDOWNSCALE BRANCH ------------------------------- ↓ #

    adk = (10, 10)
    uk1, uk2, uk3, uk4 = (25, 25), (19, 19), (13, 13), (9, 9)
    dk1, dk2, dk3, dk4 = (9, 9), (7, 7), (5, 5), (3, 3)
    fu1 = (last(afu4)+3)=>32
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

    admm1 = ADMMDeconv(adk, 50, relu, iso=true)
    ud_1 = updownblock(uk1, dk1, fu1, fd1, imshape)
    ud_2 = updownblock(uk2, dk2, fu2, fd2, imshape)
    ud_3 = updownblock(uk3, dk3, fu3, fd3, imshape)
    ud_4 = updownblock(uk4, dk4, fu4, fd4, imshape)
    ud_5 = updownblock(uk4, dk4, fu5, fd5, imshape)
    ud_6 = updownblock(uk4, dk4, fu6, fd6, imshape)

    skip_ud_34 = SkipConnection(Chain(ud_3, ud_4), +)
    skip_ud_2345 = SkipConnection(Chain(ud_2, skip_ud_34, ud_5), chcat)
    updownscale = Chain(admm1, ud_1, skip_ud_2345, ud_6)

    # ↑ -------------------------------UPDOWNSCALE BRANCH ------------------------------- ↑ #


    # ↓ -------------------------------NxDECONVS BRANCH ------------------------------- ↓ #

    k1, k2, k3 = (7, 7), (10, 10), (15, 15)
    λ1, λ2, λ3 = 0.004f0, 0.04f0, 0.4f0
    ρ1, ρ2, ρ3 = 0.02f0, 0.04f0, 0.06f0
    deconv_1 = ADMMDeconvF3(k1, 50, λ1, ρ1, relu6, iso=mcfg["use_iso"])
    deconv_2 = ADMMDeconvF3(k2, 50, λ2, ρ2, relu6, iso=mcfg["use_iso"])
    deconv_3 = ADMMDeconvF3(k3, 50, λ3, ρ3, relu6, iso=!mcfg["use_iso"])
    deconv_4 = ADMMDeconv((), 50, relu6, iso=mcfg["use_iso"])
    deconv_5 = ADMMDeconv((), 50, relu6, iso=mcfg["use_iso"])
    deconv_6 = ADMMDeconv((), 50, relu6, iso=!mcfg["use_iso"])

    deconvs = Parallel(
        chcat, 
        deconv_1, deconv_2,
        deconv_3, deconv_4, 
        deconv_5, deconv_6
        )

    # updown = updownblock(uk3, dk3, 18=>18, 18=>18, imshape)

    # ↑ -------------------------------NxDECONVS BRANCH ------------------------------- ↑ #


    residual_autoencoder = SkipConnection(autoencoder, chcat)
    auto_and_ud = Chain(residual_autoencoder, updownscale)

    auto_ud_deconv = Parallel(chcat, auto_and_ud, deconvs)

    last_updown = updownblock(uk3, dk3, (last(fd6)+18)=>32, 32=>3, imshape)

    fin = Chain(auto_ud_deconv, last_updown)

    paramCount = 0
    for layer in fin
        paramCount += sum(length, Flux.params(layer), init=0)
    end

    printstyled("\n\nMODEL SIZE (#parameters): $paramCount", bold=true, italic=true, color=:light_magenta)

    return fin
end
