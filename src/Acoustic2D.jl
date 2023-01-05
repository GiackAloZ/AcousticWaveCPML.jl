module Acoustic2D

    @doc raw"""
    solve2D(
        lx::Real,
        ly::Real,
        lt::Real,
        vel::Matrix{<:Real},
        srcs::Sources,
        recs::Receivers;
        halo::Integer = 20,
        rcoef::Real = 0.0001,
        ppw::Real = 10.0,
        freetop::Bool = true,
        do_bench::Bool = false,
        do_vis::Bool = false,
        nvis::Integer = 5,
        gif_name::String = "acoustic2D",
        plims::Vector{<:Real} = [-1.0, 1.0],
        threshold::Real = 0.01
    )

    Solve 2D acoustic wave equation on a model with size `lx`x`ly` [m], for final time `lt` [sec] with the velocity model `vel` [m/s].
    The size of `vel` implicitly specifies the number of grid points to use for the simulation. 

    Sources and receivers are specified by `srcs` and `recs`.

    Return the final time pressure field as a vector and populate the receivers seismograms with the recorded traces.

    # Arguments
    - `halo::Integer = 20`: the number of CPML layers.
    - `rcoef::Real = 0.0001`: the reflection coeffiecient for CPML layers.
    - `ppw::Real = 10.0`: minimum number of points per wavelength admissible.
    - `freetop::Bool = true`: enables free top boundary conditions.
    - `do_bench::Bool = false`: do benchmark instead of computation.
    - `do_vis::Bool = false`: enables plotting.
    - `nvis::Integer = 5`: plotting time step frequency.
    - `gif_name::String = "acoustic1D"`: name of animation plot.
    - `plims::Vector{<:Real} = [-1.0, 1.0]`: limits for pressure values in plot.
    - `threshold::Real = 0.01`: percentage of `plims` to use as threshold for plots.
    """
    solve2D()

    include(joinpath("shared", "acoustic_2D.jl"))

    export solve2D
end