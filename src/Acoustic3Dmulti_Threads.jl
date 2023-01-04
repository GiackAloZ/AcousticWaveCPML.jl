module Acoustic3Dmulti_Threads

    @doc raw"""
    TODO
    """
    solve3D_multi()

    using ImplicitGlobalGrid
    import MPI

    using ParallelStencil
    using ParallelStencil.FiniteDifferences3D
    @init_parallel_stencil(Threads, Float64, 3)

    include(joinpath("shared", "acoustic_3D_multixPU.jl"))

    export solve3D_multi
end