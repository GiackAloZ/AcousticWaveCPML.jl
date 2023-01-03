module Acoustic1D

    @doc raw"""
        solve1D(
            lx::Real,
            lt::Real,
            vel::Vector{<:Real},
            srcs::Sources,
            recs::Receivers;
            halo::Integer = 20,
            rcoef::Real = 0.0001,
            ppw::Real = 10.0,
            do_bench::Bool = false,
            do_vis::Bool = false,
            nvis::Integer = 5,
            gif_name::String = "acoustic1D",
            plims::Tuple{<:Real, <:Real} = (-1.0, 1.0)
        )

    Compute `nt` timesteps of the acoustic 1D wave equation with CPML boundary conditions on a model with size `lx` meters,
    velocity field `vel`, number of CPML layers in each boundary `halo` and CPML reflection coeffiecient `rcoef`.

    Return the last timestep pressure.

    # Arguments
    - `do_vis`: to plot visualization or not.
    - `do_bench`: to perform a benchmark instead of the computation.
    - `nvis`: frequency of timestep for visualization
    - `gif_name`: name of the gif to save
    - `plims`: pressure limits in visualizion plot
    """
    solve1D()

    include(joinpath("shared", "acoustic_1D.jl"))

    export solve1D
end