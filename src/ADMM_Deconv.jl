module ADMM_Deconv


using Flux, CUDA, TestImages, Noise, IterTools, ProgressBars, Zygote

include("layers/deconv_admm.jl")
include("utilities/base_funcs.jl")
include("processing/datafeeder.jl")
include("metrics/iqi.jl")

# Avoid Zygote issue with no gradients defined for primitive CUDA kernels such as CUDA.ones / CUDA.zeros
Zygote.@nograd CUDA.ones
Zygote.@nograd CUDA.zeros

# plotlyjs()

train_set = ImageDataFeeder("D:/Projects/ISETC2022/dcnn-deblur/dataset/GOPRO_Large/train/origblur/x_set", "D:/Projects/ISETC2022/dcnn-deblur/dataset/GOPRO_Large/train/y_set", ".png", (32, 32), (32, 32))

traintest = Flux.DataLoader(train_set, batchsize=2)|>gpu

model_test = Chain(
  ADMMDeconv((32,32), 50, relu6), 
  Conv((1,1), 3=>3, relu6))|>gpu

# @show model(traintest.data[1][1])

# evalcb = () -> @show(loss(img_n, data_arr))
ps = Flux.params(model_test)
opt = Flux.ADAM(0.0005)

loss(x, y) = ssim_loss(x, y)|>gpu

metric(x, y) = psnr(x, y)|>gpu


for epoch in 1:2
  for (x,y) in ProgressBar(traintest)
    out = model_test(x)
    gs = Flux.gradient(() -> loss(out, y), ps)
    Flux.update!(opt,ps,gs)

    print("Loss function (SSIM)= $(loss(out, y)); PSNR= $(metric(out, y))")
  end
end


# for epoch in 1:1000
#   Flux.train!(loss_func, model_test, traintest, optim)
# end

# display(plot(x -> 2x-x^3, -2, 2, legend=false))
# display(scatter!(x -> model([x]), -2:0.1f0:2))
# sleep(10000)

end # module ADMM_Deconv
