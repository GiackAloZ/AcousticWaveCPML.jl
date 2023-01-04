module Acoustic2D_CUDA

    @doc raw"""
    TODO
    """
    solve2D()

    using CUDA
    using ParallelStencil
    using ParallelStencil.FiniteDifferences2D
    @init_parallel_stencil(CUDA, Float64, 2)

    include(joinpath("shared", "acoustic_2D_xPU.jl"))

    export solve2D
end