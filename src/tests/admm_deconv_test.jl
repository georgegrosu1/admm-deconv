using TestImages, Plots, CUDA, Images, Noise

include("../ops/ops.jl")
include("../metrics/iqi.jl")

function main()
    img1 = testimage("fabio_color_256.png")
    img2 = testimage("lena_color_256.tif")
    img3 = testimage("monarch_color_256.png")

	Images.save("orig1.png", img1)
	Images.save("orig2.png", img2)
	Images.save("orig3.png", img3)

    # display(Plots.plot(img))
    # readline()

    # blur_psf = ImageFiltering.Kernel.gaussian(1.2)
    blur_psf = zeros((7,7))
    blur_psf[4, :] .= 1/7
    test_img1 = ImageFiltering.imfilter(img1, blur_psf)
    test_img2 = ImageFiltering.imfilter(img2, blur_psf)
    test_img3 = ImageFiltering.imfilter(img3, blur_psf)

    Images.save("blurred1.png", test_img1)
	Images.save("blurred2.png", test_img2)
	Images.save("blurred3.png", test_img3)

    println(size(blur_psf))

    # display(Plots.plot(test_img1))
    # readline()
    # display(Plots.plot(test_img2))
    # readline()
    # display(Plots.plot(test_img3))
    # readline()

    nvar = 0.1
    # test_img1 = img2tensor(test_img1)
    # test_img1 .+= (nvar .* randn(eltype(test_img1), size(test_img1)))
    # clamp!(test_img1, 0.0f0, 1.0f0)
    # test_img1 = RGB.(test_img1[:,:,1], test_img1[:,:,2], test_img1[:,:,3])
    # test_img2 = img2tensor(test_img2)
    # test_img2 .+= (nvar .* randn(eltype(test_img2), size(test_img2)))
    # clamp!(test_img2, 0.0f0, 1.0f0)
    # test_img2 = RGB.(test_img2[:,:,1], test_img2[:,:,2], test_img2[:,:,3])
    # test_img3 = img2tensor(test_img3)
    # test_img3 .+= (nvar .* randn(eltype(test_img3), size(test_img3)))
    # clamp!(test_img3, 0.0f0, 1.0f0)
    # test_img3 = RGB.(test_img3[:,:,1], test_img3[:,:,2], test_img3[:,:,3])

    Images.save("blurrednoise1.png", test_img1)
	Images.save("blurrednoise2.png", test_img2)
	Images.save("blurrednoise3.png", test_img3)

    println("Blurred noise SSIM 1: ", assess_ssim(test_img1, img1))
    println("Blurred noise SSIM 2: ", assess_ssim(test_img2, img2))
    println("Blurred noise SSIM 3: ", assess_ssim(test_img3, img3))
    println("Blurred noise PSNR 1: ", assess_psnr(test_img1, img1))
    println("Blurred noise PSNR 2: ", assess_psnr(test_img2, img2))
    println("Blurred noise PSNR 3: ", assess_psnr(test_img3, img3))

    test_img1 = expand_dims(img2tensor(test_img1), 4)
    test_img2 = expand_dims(img2tensor(test_img2), 4)
    test_img3 = expand_dims(img2tensor(test_img3), 4)

    test_img = cat(test_img1, test_img2, test_img3, dims=4)
    test_ori = cat(img2tensor(img1), img2tensor(img2), img2tensor(img3), dims=4)

    println(size(test_img))

    psf_arr = parent(blur_psf)
    psf_arr = Float32.(psf_arr)[:,:,:,:]


    CUDA.@time img_restored = tvd_fft_gpu(cu(test_img), cu([0.0041f0]), cu([0.021f0]), cu(psf_arr), false)
    img_restored = Array{Float32}(img_restored)
    clamp!(img_restored, 0.0f0, 1.0f0)

    println("ADMM Test passed with no errors! ")

    # display(Plots.plot(RGB.(img_restored[:,:,1,1], img_restored[:,:,2,1], img_restored[:,:,3,1])))
    # readline()

    # display(Plots.plot(RGB.(img_restored[:,:,1,2], img_restored[:,:,2,2], img_restored[:,:,3,2])))
    # readline()

    # display(Plots.plot(RGB.(img_restored[:,:,1,3], img_restored[:,:,2,3], img_restored[:,:,3,3])))
    # readline()

    to_view1 = RGB.(img_restored[:, :, 1, 1], img_restored[:, :, 2, 1], img_restored[:, :, 3, 1])
    to_view2 = RGB.(img_restored[:, :, 1, 2], img_restored[:, :, 2, 2], img_restored[:, :, 3, 2])
    to_view4 = RGB.(img_restored[:, :, 1, 3], img_restored[:, :, 2, 3], img_restored[:, :, 3, 3])

    Images.save("restored1.png", to_view1)
	Images.save("restored2.png", to_view2)
	Images.save("restored3.png", to_view4)

    println("Restored SSIM 1: ", assess_ssim(to_view1, img1))
    println("Restored SSIM 2: ", assess_ssim(to_view2, img2))
    println("Restored SSIM 3: ", assess_ssim(to_view4, img3))
    println("Restored PSNR 1: ", assess_psnr(to_view1, img1))
    println("Restored PSNR 2: ", assess_psnr(to_view2, img2))
    println("Restored PSNR 3: ", assess_psnr(to_view4, img3))
    println(peak_snr(img_restored[:, :, :, 3], img2tensor(img3)))
    println(ssim(img_restored, test_ori))

    ff = assess_ssim(to_view1, img1) + assess_ssim(to_view2, img2) + assess_ssim(to_view4, img3)
    println(ff / 3)

end

main()