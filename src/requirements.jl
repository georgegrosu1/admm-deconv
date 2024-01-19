using Pkg
println(Pkg.status())

dependencies = [
    "ProgressBars",
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
    "ImageIO",
    "ImageFiltering",
    "CUDA",
    "cuDNN",
    "Zygote",
    "NNlib",
    "NNlibCUDA",
    "MLUtils",
    "Flux",
    "DeconvOptim",
    "BSON"
]

Pkg.add(dependencies)
# Pkg.rm(dependencies)