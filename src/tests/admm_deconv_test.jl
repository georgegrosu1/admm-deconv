using TestImages, Plots

include("../ops/ops.jl")

function main()
    img = testimage("fabio_color_256.png")
    img_ref = img2tensor(img)

    # display(Plots.plot(img))
    # readline()

    blur_psf = ImageFiltering.Kernel.gaussian(2)
    test_img = ImageFiltering.imfilter(img, blur_psf)


    test_img = img2tensor(test_img)
    test_img = test_img + 0.f05*randn(eltype(test_img), size(test_img))

    to_view = RGB.(test_img[:, :, 1], test_img[:, :, 2], test_img[:, :, 3])

    test_img = expand_dims(test_img, 4)

    psf_arr = parent(blur_psf)
    psf_arr = Float32.(psf_arr)[:,:,:,:]


    @time img_restored = tvd_fft(test_img, 0.9f-2, 0.5f0; h=psf_arr, isotropic=true)

    println("ADMM Test passed with no errors! ")

    display(Plots.plot(RGB.(img_restored[:,:,1,1], img_restored[:,:,2,1], img_restored[:,:,3,1])))

    readline()

    to_view = RGB.(img_restored[:, :, 1], img_restored[:, :, 2], img_restored[:, :, 3])
    ref = RGB.(img_ref[:, :, 1], img_ref[:, :, 2], img_ref[:, :, 3])

    println(assess_ssim(to_view, ref))
end

main()