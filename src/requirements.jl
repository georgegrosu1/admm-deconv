using Pkg
println(Pkg.status())

dependencies = [
    "Glob",
    "IterTools",
    "Plots",
    "LinearAlgebra",
    "PaddedViews",
    "FFTW",
    "Pipe",
    "DataStructures",
    "PlotlyJS",
    "PyPlot",
    "VegaLite",
    "FileIO",
    "Colors",
    "ColorVectorSpace",
    "ColorTypes",
    "ImageCore",
    "Images",
    "ImageShow",
    "ImageView",
    "TestImages",
    "ImageTransformations",
    "Noise",
    "DSP",
    "FixedPointNumbers",
    "DeconvOptim",
    "ImageIO",
    "ImageFiltering",
    "cuDNN",
    "CUDA",
    "Zygote",
    "NNlib",
    "Flux"
]

Pkg.add(dependencies)
# Pkg.rm(dependencies)