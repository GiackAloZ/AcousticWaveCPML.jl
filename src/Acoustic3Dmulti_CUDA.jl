@doc raw"""
Module with acoustic 3D multi-xPU solver using `CUDA`.
"""
module Acoustic3Dmulti_CUDA

    @doc raw"""
        solve3D_multi(
            lx::Real,
            ly::Real,
            lz::Real,
            lt::Real,
            nx::Integer,
            ny::Integer,
            nz::Integer,
            vel_func::Function,
            srcs::Sources,
            recs::Receivers;
            halo::Integer = 20,
            rcoef::Real = 0.0001,
            ppw::Real = 10.0,
            freetop::Bool = true,
            do_vis::Bool = false,
            do_save::Bool = false,
            nvis::Integer = 5,
            nsave::Integer = 5,
            gif_name::String = "acoustic3D_multixPU_slice",
            save_name::String = "acoustic3D_multixPU",
            plims::Vector{<:Real} = [-1.0, 1.0],
            threshold::Real = 0.01,
            init_MPI::Bool = true
        )
    
    Solve 3D acoustic wave equation on multiple MPI processes on a model with size `lx`x`ly`x`lz` [m], for final time `lt` [sec] with the velocity model `vel_func` [m/s].
    The LOCAL number of grid points to use for the simulation (for each MPI processes) is specified by `nx`, `ny` and `nz`.
    The velocity model `vel_func` must be a function accepting three arguments, the x, y and z coordinates of a point in meters, and returning the velocity value at that point.

    Sources and receivers are specified by `srcs` and `recs`.

    Return the LOCAL final time pressure field as a vector and populate the receivers seismograms with the LOCAL recorded traces.
    This means that the each MPI process returns only its piece of final pressure field and only records traces of receivers inside of its local domain.
    
    If `do_vis`, create and save visualization in `simulations` folder.
    If `do_save`, create and save intermediate pressure fields in `simulations/tmp` folder.
    
    # Arguments
    - `halo::Integer = 20`: the number of CPML layers.
    - `rcoef::Real = 0.0001`: the reflection coeffiecient for CPML layers.
    - `ppw::Real = 10.0`: minimum number of points per wavelength admissible.
    - `freetop::Bool = true`: enables free top boundary conditions.
    - `do_vis::Bool = false`: enables plotting.
    - `do_save::Bool = false`: enables saving intermediate pressure fields.
    - `nvis::Integer = 5`: plotting time step frequency.
    - `nsave::Integer = 5`: saving time step frequency.
    - `gif_name::String = "acoustic3D_multixPU_slice"`: name of animation plot.
    - `save_name::String = "acoustic3D_multixPU"`: prefix names of saved pressure files.
    - `plims::Vector{<:Real} = [-1.0, 1.0]`: limits for pressure values in plot.
    - `threshold::Real = 0.01`: percentage of `plims` to use as threshold for plots
    - `init_MPI::Bool = true`: initialize MPI with ImplicitGlobalGrid.
    """
    solve3D_multi()

    using ImplicitGlobalGrid
    import MPI

    using CUDA
    using ParallelStencil
    using ParallelStencil.FiniteDifferences3D
    @init_parallel_stencil(CUDA, Float64, 3)

    include(joinpath("shared", "acoustic_3D_multixPU.jl"))

    export solve3D_multi
end