@doc raw"""
Module with acoustic 3D xPU solver using `Base.Threads`.
"""
module Acoustic3D_Threads

    @doc raw"""
        solve3D(
            lx::Real,
            ly::Real,
            lz::Real,
            lt::Real,
            vel::Array{<:Real, 3},
            srcs::Sources,
            recs::Receivers;
            halo::Integer = 20,
            rcoef::Real = 0.0001,
            ppw::Real = 10.0,
            freetop::Bool = true,
            do_bench::Bool = false,
            do_vis::Bool = false,
            do_save::Bool = false,
            nvis::Integer = 5,
            nsave::Integer = 5,
            gif_name::String = "acoustic3D_xPU_slice",
            save_name::String = "acoustic3D_xPU",
            plims::Vector{<:Real} = [-1.0, 1.0],
            threshold::Real = 0.01
        )

    Solve 3D acoustic wave equation on a model with size `lx`x`ly`x`lz` [m], for final time `lt` [sec] with the velocity model `vel` [m/s].
    The size of `vel` implicitly specifies the number of grid points to use for the simulation. 

    Sources and receivers are specified by `srcs` and `recs`.

    Return the final time pressure field as a vector and populate the receivers seismograms with the recorded traces.

    If `do_vis`, create and save visualization in `simulations` folder.
    If `do_save`, create and save intermediate pressure fields in `simulations/tmp` folder.

    # Arguments
    - `halo::Integer = 20`: the number of CPML layers.
    - `rcoef::Real = 0.0001`: the reflection coeffiecient for CPML layers.
    - `ppw::Real = 10.0`: minimum number of points per wavelength admissible.
    - `freetop::Bool = true`: enables free top boundary conditions.
    - `do_bench::Bool = false`: do benchmark instead of computation.
    - `do_vis::Bool = false`: enables plotting.
    - `do_save::Bool = false`: enables saving intermediate pressure fields.
    - `nvis::Integer = 5`: plotting time step frequency.
    - `nsave::Integer = 5`: saving time step frequency.
    - `gif_name::String = "acoustic3D_xPU_slice"`: name of animation plot.
    - `save_name::String = "acoustic3D_xPU"`: prefix names of saved pressure files.
    - `plims::Vector{<:Real} = [-1.0, 1.0]`: limits for pressure values in plot.
    - `threshold::Real = 0.01`: percentage of `plims` to use as threshold for plots.
    """
    solve3D()

    using ParallelStencil
    using ParallelStencil.FiniteDifferences3D
    @init_parallel_stencil(Threads, Float64, 3)

    include(joinpath("shared", "acoustic_3D_xPU.jl"))

    export solve3D
end