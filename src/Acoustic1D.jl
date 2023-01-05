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
            plims::Vector{<:Real} = [-1.0, 1.0]
        )

    Solve 1D acoustic wave equation on a model with length `lx` [m], for final time `lt` [sec] with the velocity model `vel` [m/s].
    The size of `vel` implicitly specifies the number of grid points to use for the simulation. 

    Sources and receivers are specified by `srcs` and `recs`.

    Return the final time pressure field as a vector and populate the receivers seismograms with the recorded traces.

    If `do_vis`, create and save visualization in `simulations` folder.

    # Arguments
    - `halo::Integer = 20`: the number of CPML layers.
    - `rcoef::Real = 0.0001`: the reflection coeffiecient for CPML layers.
    - `ppw::Real = 10.0`: minimum number of points per wavelength admissible.
    - `do_bench::Bool = false`: do benchmark instead of computation.
    - `do_vis::Bool = false`: enables plotting.
    - `nvis::Integer = 5`: plotting time step frequency.
    - `gif_name::String = "acoustic1D"`: name of animation plot.
    - `plims::Vector{<:Real} = [-1.0, 1.0]`: limits for pressure values in plot.
    """
    solve1D()

    include(joinpath("shared", "acoustic_1D.jl"))

    export solve1D
end