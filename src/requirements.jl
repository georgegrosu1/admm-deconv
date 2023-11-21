using Pkg
println(Pkg.status())

dependencies = [
    "Plots",
    "LinearAlgebra",
    "PaddedViews",
    "FFTW",
    "DataStructures",
    "PlotlyJS",
    "PyPlot",
    "VegaLite",
    "FileIO",
    "ColorVectorSpace",
    "ColorTypes",
    "Images",
    "ImageShow",
    "ImageView",
    "ImageTransformations",
    "DSP",
    "FixedPointNumbers",
    "ImageIO",
    "ImageFiltering",
    "cuDNN",
    "CUDA",
    "Flux"
]

Pkg.add(dependencies)
# Pkg.rm(dependencies)