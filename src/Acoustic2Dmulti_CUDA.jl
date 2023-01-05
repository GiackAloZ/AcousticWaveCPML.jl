module Acoustic2Dmulti_CUDA

    @doc raw"""
    solve2D_multi(
        lx::Real,
        ly::Real,
        lt::Real,
        nx::Integer,
        ny::Integer,
        vel_func::Function,
        srcs::Sources,
        recs::Receivers;
        halo::Integer = 20,
        rcoef::Real = 0.0001,
        ppw::Real = 10.0,
        freetop::Bool = true,
        do_vis::Bool = false,
        nvis::Integer = 5,
        gif_name::String = "acoustic2D_multixPU",
        plims::Vector{<:Real} = [-1, 1],
        threshold::Real = 0.01,
        init_MPI::Bool = true
    )

    Solve 2D acoustic wave equation on multiple MPI processes on a model with size `lx`x`ly` [m], for final time `lt` [sec] with the velocity model `vel_func` [m/s].
    The LOCAL number of grid points to use for the simulation (for each MPI processes) is specified by `nx` and `ny`.
    The velocity model `vel_func` must be a function accepting two arguments, the x and y coordinates of a point in meters, and returning the velocity value at that point.

    Sources and receivers are specified by `srcs` and `recs`.

    Return the LOCAL final time pressure field as a vector and populate the receivers seismograms with the LOCAL recorded traces.
    This means that the each MPI process returns only its piece of final pressure field and only records traces of receivers inside of its local domain.

    If `do_vis`, create and save visualization in `simulations` folder.
    
    # Arguments
    - `halo::Integer = 20`: the number of CPML layers.
    - `rcoef::Real = 0.0001`: the reflection coeffiecient for CPML layers.
    - `ppw::Real = 10.0`: minimum number of points per wavelength admissible.
    - `freetop::Bool = true`: enables free top boundary conditions.
    - `do_vis::Bool = false`: enables plotting.
    - `nvis::Integer = 5`: plotting time step frequency.
    - `gif_name::String = "acoustic2D_multixPU"`: name of animation plot.
    - `plims::Vector{<:Real} = [-1.0, 1.0]`: limits for pressure values in plot.
    - `threshold::Real = 0.01`: percentage of `plims` to use as threshold for plots
    - `init_MPI::Bool = true`: initialize MPI with ImplicitGlobalGrid.
    """
    solve2D_multi()

    using ImplicitGlobalGrid
    import MPI

    using CUDA
    using ParallelStencil
    using ParallelStencil.FiniteDifferences2D
    @init_parallel_stencil(CUDA, Float64, 2)

    include(joinpath("shared", "acoustic_2D_multixPU.jl"))

    export solve2D_multi
end