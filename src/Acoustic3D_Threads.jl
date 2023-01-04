module Acoustic3D_Threads

    @doc raw"""
    TODO
    """
    solve3D()

    using ParallelStencil
    using ParallelStencil.FiniteDifferences3D
    @init_parallel_stencil(Threads, Float64, 3)

    include(joinpath("shared", "acoustic_3D_xPU.jl"))

    export solve3D
end