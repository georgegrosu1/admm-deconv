module ADMM_Deconv

using Flux, Plots, CUDA, TestImages, DSP, Noise, IterTools
include("layers/deconv_admm.jl")
include("utilities/base_funcs.jl")

plotlyjs()


data = testimage("mandril_color.tif")
data_arr = img2tensor(data)
data_arr = expand_dims(data_arr, 4)

psf = Float32.(generate_psf((size(data_arr)[1],size(data_arr)[2]), 3))
psf = expand_dims(psf, 3)
img_b = DeconvOptim.conv(data_arr, psf)
img_n = Noise.poisson(img_b, 300)

traintest = Flux.DataLoader((data=img_n, label=data_arr), batchsize=1) 
@show size(traintest.data[1])

model = Chain(Conv((9,9), 3=>3, Flux.relu), Flux.relu)

model(traintest.data[1])

loss_func(x, y) = Flux.mse.(model(x), y)
evalcb = () -> @show(loss(img_n, data_arr))
ps = Flux.params(model)
@show ps

optim = Flux.setup(Adam(0.005), model)
for epoch in 1:1000
  Flux.train!(loss_func, model, IterTools.ncycle(traintest, 10), optim)
end

# display(plot(x -> 2x-x^3, -2, 2, legend=false))
# display(scatter!(x -> model([x]), -2:0.1f0:2))
# sleep(10000)

end # module ADMM_Deconv
