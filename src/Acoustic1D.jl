module Acoustic1D

    @doc raw"""
        solve1D(
            lx::Real,
            nt::Integer,
            vel::Vector{<:Real};
            halo::Integer = 20,
            rcoef::Real = 0.0001,
            do_vis::Bool = true,
            do_bench::Bool = false,
            nvis::Integer = 2,
            gif_name::String = "acoustic1D",
            plims::Vector{<:Real} = [-1e-1, 1e-1]
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