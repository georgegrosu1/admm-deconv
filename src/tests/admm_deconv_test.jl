using TestImages, Plots

include("../ops/ops.jl")

function main()
    test_img = testimage("mandrill.tiff")

    blur_psf = ImageFiltering.Kernel.gaussian(5)
    test_img = imfilter(test_img, blur_psf)
    
    test_img = img2tensor(test_img)
    test_img = expand_dims(test_img, 4)
    psf_arr = expand_dims(expand_dims(parent(blur_psf), 3), 4)
    psf_arr = Float32.(psf_arr)

    admmSet = ADMMSettings()

    @time img_restored = admm_tv_deconv(test_img, psf_arr, Int32(10), 1000f0, admmSet)

    println("ADMM Test passed with no errors!")

    to_view = RGB.(img_restored[:, :, 1, 1], img_restored[:, :, 2, 1], img_restored[:, :, 3, 1])

    display(Plots.plot(to_view))

    sleep(100000)
end

main()