module ADMM_Deconv

using Flux, Plots, CUDA, TestImages, DSP, Noise, IterTools, ProgressBars
include("layers/deconv_admm.jl")
include("utilities/base_funcs.jl")
include("processing/datafeeder.jl")

plotlyjs()

train_set = ImageDataFeeder("D:/Projects/ISETC2022/dcnn-deblur/dataset/GOPRO_Large/train/x_set", "D:/Projects/ISETC2022/dcnn-deblur/dataset/GOPRO_Large/train/y_set", ".png", (16, 16), (16, 16), 8)

traintest = Flux.DataLoader(train_set, batchsize=4)

model_test = Chain(ADMMDeconv((1,1), 3=>3, relu), Conv((1,1), 3 => 3, relu))

# @show model(traintest.data[1][1])

# evalcb = () -> @show(loss(img_n, data_arr))
ps = Flux.params(model_test)
opt = Flux.ADAM(0.0005)

loss(x, y) = Flux.mse(model_test(x), y)


for epoch in 1:2
  for (x,y) in ProgressBar(traintest)
    gs = Flux.gradient(() -> loss(x, y), ps)
    Flux.update!(opt,ps,gs)
  end
end


# for epoch in 1:1000
#   Flux.train!(loss_func, model_test, traintest, optim)
# end

# display(plot(x -> 2x-x^3, -2, 2, legend=false))
# display(scatter!(x -> model([x]), -2:0.1f0:2))
# sleep(10000)

end # module ADMM_Deconv
