using TestImages, Plots, CUDA

include("../ops/ops.jl")
include("../metrics/iqi.jl")

function main()
    img1 = testimage("fabio_color_256.png")
    img2 = testimage("lena_color_256.tif")
    img3 = testimage("monarch_color_256.png")

    # display(Plots.plot(img))
    # readline()

    blur_psf = ImageFiltering.Kernel.gaussian(1.2)
    # blur_psf = zeros((8,8))
    # blur_psf[4, :] .= 1
    # blur_psf ./= 8
    test_img1 = ImageFiltering.imfilter(img1, blur_psf)
    test_img2 = ImageFiltering.imfilter(img2, blur_psf)
    test_img3 = ImageFiltering.imfilter(img3, blur_psf)

    println(size(blur_psf))

    # display(Plots.plot(test_img1))
    # readline()
    # display(Plots.plot(test_img2))
    # readline()
    # display(Plots.plot(test_img3))
    # readline()

    test_img1 = img2tensor(test_img1)
    test_img1 = test_img1 + 0.f05*randn(eltype(test_img1), size(test_img1))
    test_img2 = img2tensor(test_img2)
    test_img2 = test_img2 + 0.f05*randn(eltype(test_img2), size(test_img2))
    test_img3 = img2tensor(test_img3)
    test_img3 = test_img3 + 0.f05*randn(eltype(test_img3), size(test_img3))

    test_img1 = expand_dims(test_img1, 4)
    test_img2 = expand_dims(test_img2, 4)
    test_img3 = expand_dims(test_img3, 4)

    test_img = cat(test_img1, test_img2, test_img3, dims=4)
    test_ori = cat(img2tensor(img1), img2tensor(img2), img2tensor(img3), dims=4)

    println(size(test_img))

    psf_arr = parent(blur_psf)
    psf_arr = Float32.(psf_arr)[:,:,:,:]


    @time img_restored = tvd_fft_gpu(cu(test_img), cu([0.9f-2]), cu([0.005f0]), cu(psf_arr), true)
    img_restored = Array{Float32}(img_restored)
    clamp!(img_restored, 0.0f0, 1.0f0)

    println("ADMM Test passed with no errors! ")

    display(Plots.plot(RGB.(img_restored[:,:,1,1], img_restored[:,:,2,1], img_restored[:,:,3,1])))
    readline()

    display(Plots.plot(RGB.(img_restored[:,:,1,2], img_restored[:,:,2,2], img_restored[:,:,3,2])))
    readline()

    display(Plots.plot(RGB.(img_restored[:,:,1,3], img_restored[:,:,2,3], img_restored[:,:,3,3])))
    readline()

    to_view1 = RGB.(img_restored[:, :, 1, 1], img_restored[:, :, 2, 1], img_restored[:, :, 3, 1])
    to_view2 = RGB.(img_restored[:, :, 1, 2], img_restored[:, :, 2, 2], img_restored[:, :, 3, 2])
    to_view4 = RGB.(img_restored[:, :, 1, 3], img_restored[:, :, 2, 3], img_restored[:, :, 3, 3])

    println(assess_ssim(to_view1, img1))
    println(assess_ssim(to_view2, img2))
    println(assess_ssim(to_view4, img3))
    println(assess_psnr(to_view4, img3))
    println(peak_snr(img_restored[:, :, :, 3], img2tensor(img3)))
    println(ssim(img_restored, test_ori))

    ff = assess_ssim(to_view1, img1) + assess_ssim(to_view2, img2) + assess_ssim(to_view4, img3)
    println(ff / 3)

end

main()