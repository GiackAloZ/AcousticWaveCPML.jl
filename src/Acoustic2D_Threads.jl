module Acoustic2D_Threads

    @doc raw"""
        solve2D(
            lx::Real,
            ly::Real,
            nt::Integer,
            vel::Matrix{<:Real},
            possrcs;
            halo::Integer = 20,
            rcoef::Real = 0.0001,
            do_vis::Bool = true,
            do_bench::Bool = false,
            nvis::Integer = 5,
            gif_name::String = "acoustic2D_xPU",
            plims::Vector{<:Real} = [-3, 3],
            threshold::Real = 0.01,
            freetop::Bool = true
        )

    Compute `nt` timesteps of the acoustic 2D wave equation using ParallelStencil on xPUs (with `Base.Threads`) with CPML boundary conditions on a model with size `lx`x`ly` meters,
    velocity field `vel`, position of sources `possrcs`, number of CPML layers in each boundary `halo` and CPML reflection coeffiecient `rcoef`.

    The position of sources must be a 2D array with the `size(possrcs,1)` equal to the number of sources and `size(possrcs,2)` equal to 2.

    Return the last timestep pressure.

    # Arguments
    - `do_vis`: to plot visualization or not.
    - `do_bench`: to perform a benchmark instead of the computation.
    - `nvis`: frequency of timestep for visualization
    - `gif_name`: name of the gif to save
    - `plims`: pressure limits in visualizion plot
    - `threshold`: percentage of `plims` to cut out of visualization.
    - `freetop`: to have free top BDCs or not.
    """
    solve2D()

    using ParallelStencil
    using ParallelStencil.FiniteDifferences2D
    @init_parallel_stencil(Threads, Float64, 2)

    include(joinpath("shared", "acoustic_2D_xPU.jl"))

    export solve2D
end