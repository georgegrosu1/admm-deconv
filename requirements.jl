using Pkg
println(Pkg.status())

dependencies = [
    "ArgParse",
    "JSON",
    "ProgressBars",
    "Glob",
    "IterTools",
    "Plots",
    "LinearAlgebra",
    "PaddedViews",
    "FFTW",
    "Pipe",
    "DataStructures",
    "PyPlot",
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
    "Statistics",
    "Noise",
    "DSP",
    "FixedPointNumbers",
    "ImageIO",
    "ImageFiltering",
    "CUDA",
    "cuDNN",
    "ChainRulesCore",
    "Zygote",
    "NNlib", 
    "NNlibCUDA",
    "MLUtils",
    "Flux",
    "DeconvOptim",
    "JLD2"
]

Pkg.add(dependencies)
# Pkg.rm(dependencies)