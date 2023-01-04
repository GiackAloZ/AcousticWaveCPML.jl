module Acoustic2Dmulti_Threads

    @doc raw"""
    TODO
    """
    solve2D_multi()

    using ImplicitGlobalGrid
    import MPI

    using ParallelStencil
    using ParallelStencil.FiniteDifferences2D
    @init_parallel_stencil(Threads, Float64, 2)

    include(joinpath("shared", "acoustic_2D_multixPU.jl"))

    export solve2D_multi
end