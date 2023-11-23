using TestImages

include("../ops/ops.jl")

function main()
    test_img = testimage("mandrill.tiff")
    test_img = img2tensor(test_img)
    test_img = expand_dims(test_img, 4)

    admmSet = ADMMSettings()
    kern = Float32.([[1 0 1]; [0 1 0]; [1 0 1]])
    kern = expand_dims(expand_dims(kern, 3), 4)

    @time img_restored = admm_tv_deconv(test_img, kern, Int32(20), Float32(0.6), admmSet)

    println("ADMM Test passed with no errors!")
end

main()