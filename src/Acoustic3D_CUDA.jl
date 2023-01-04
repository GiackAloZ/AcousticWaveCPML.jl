module Acoustic3D_CUDA

    @doc raw"""
    TODO
    """
    solve3D()

    using CUDA
    using ParallelStencil
    using ParallelStencil.FiniteDifferences3D
    @init_parallel_stencil(CUDA, Float64, 3)

    include(joinpath("shared", "acoustic_3D_xPU.jl"))

    export solve3D
end