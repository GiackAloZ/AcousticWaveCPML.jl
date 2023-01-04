using Test, ReferenceTests, BSON
using DSP, NumericalIntegration, LinearAlgebra

using AcousticWaveCPML
using AcousticWaveCPML.Acoustic2Dmulti_Threads

REF_FLD = joinpath(@__DIR__, "references")

comp(d1, d2) = keys(d1)==keys(d2) &&
        all([ v1 ≈ v2 for (v1,v2) in zip(values(d1), values(d2))])

import MPI
MPI.Init()

@testset "Test homogeneous velocity analytical solution" begin
    # simple constant velocity model
    nx = ny = 501                           # grid size
    lx = ly = 10.0                          # model sizes [m]
    c0 = 1.0
    vel_func(x,y) = c0                      # velocity model [m/s]
    lt = 10.0                               # final time [s]
    # sources
    f0 = 1.0                                # source dominating frequency [Hz]
    t0 = 4 / f0                             # source activation time [s]
    stf = rickersource1D                    # second derivative of gaussian
    possrcs = zeros(1,2)
    possrcs[1,:] .= [lx/2, ly/2]
    srcs = Sources(possrcs, [t0], [stf], f0)
    # receivers
    posrecs = zeros(1,2)
    posrecs[1,:] .= [lx/4, ly/2]
    recs = Receivers(posrecs)

    # Numerical solution
    solve2D_multi(lx, ly, lt, nx, ny, vel_func, srcs, recs; halo=0, freetop=false, init_MPI=false)
    numerical_trace = recs.seismograms[:,1]

    # Analytical solution
    dx = lx / (nx-1)
    dy = ly / (ny-1)
    dt = sqrt(2) / (c0 * (1/dx + 1/dy)) / 2
    nt = ceil(Int, lt / dt)
    times = collect(range(0.0, step=dt, length=nt+1))
    dist = norm(possrcs[1,:] .- posrecs[1,:])
    src = rickersource1D.(times, t0, f0)
    # Calculate Green's function
    G = times .* 0.
    for it = 1:nt+1
        # Heaviside function
        if (times[it] - dist / c0) >= 0
            G[it] = 1. / (2π * c0^2 * sqrt((times[it]^2) - (dist^2 / (c0^2))))
        end
    end
    # Convolve with source term
    Gc = conv(G, src .* dt)
    Gc = Gc[2:nt+1]

    @test length(numerical_trace) == length(Gc) == nt
    # test integral of absolute difference over time is less then a constant 1% error relative to the peak analytical solution
    @test integrate(times[2:end], abs.(numerical_trace .- Gc)) <= maximum(abs.(Gc)) * 0.01 * lt
end

@testset "Test homogeneous velocity analytical solution CPML halo 20" begin
    # simple constant velocity model
    nx = ny = 501                           # grid size
    lx = ly = 10.0                          # model sizes [m]
    c0 = 1.0
    vel_func(x,y) = c0                      # velocity model [m/s]
    lt = 20.0                               # final time [s]
    # sources
    f0 = 1.0                                # source dominating frequency [Hz]
    t0 = 4 / f0                             # source activation time [s]
    stf = rickersource1D                    # second derivative of gaussian
    possrcs = zeros(1,2)
    possrcs[1,:] .= [lx/2, ly/2]
    srcs = Sources(possrcs, [t0], [stf], f0)
    # receivers
    posrecs = zeros(1,2)
    posrecs[1,:] .= [lx/4, ly/2]
    recs = Receivers(posrecs)

    # Numerical solution
    solve2D_multi(lx, ly, lt, nx, ny, vel_func, srcs, recs; halo=20, freetop=false, init_MPI=false)
    numerical_trace = recs.seismograms[:,1]

    # Analytical solution
    dx = lx / (nx-1)
    dy = ly / (ny-1)
    dt = sqrt(2) / (c0 * (1/dx + 1/dy)) / 2
    nt = ceil(Int, lt / dt)
    times = collect(range(0.0, step=dt, length=nt+1))
    dist = norm(possrcs[1,:] .- posrecs[1,:])
    src = rickersource1D.(times, t0, f0)
    # Calculate Green's function
    G = times .* 0.
    for it = 1:nt+1
        # Heaviside function
        if (times[it] - dist / c0) >= 0
            G[it] = 1. / (2π * c0^2 * sqrt((times[it]^2) - (dist^2 / (c0^2))))
        end
    end
    # Convolve with source term
    Gc = conv(G, src .* dt)
    Gc = Gc[2:nt+1]

    @test length(numerical_trace) == length(Gc) == nt
    # test integral of absolute difference over time is less then a constant 1% error relative to the peak analytical solution
    @test integrate(times[2:end], abs.(numerical_trace .- Gc)) <= maximum(abs.(Gc)) * 0.01 * lt
end

@testset "Test reference constant velocity" begin
    # simple constant velocity model
    nx = ny = 201                           # grid size
    lx = ly = 2000.0                        # model sizes [m]
    vel_func(x,y) = 2000.0                  # velocity model [m/s]
    lt = 2.0                                # final time [s]
    # sources
    f0 = 10.0                               # source dominating frequency [Hz]
    t0 = 4 / f0                             # source activation time [s]
    stf = rickersource1D                    # second derivative of gaussian
    possrcs = zeros(1,2)
    possrcs[1,:] .= [lx/2, ly/2]
    srcs = Sources(possrcs, [t0], [stf], f0)
    # receivers
    posrecs = zeros(2,2)
    posrecs[1,:] .= [lx/2,  2ly/3]
    posrecs[2,:] .= [2lx/3, 2ly/3]
    recs = Receivers(posrecs)

    pend = solve2D_multi(lx, ly, lt, nx, ny, vel_func, srcs, recs;
            halo=20, rcoef=0.0001, do_vis=false, freetop=false, init_MPI=false)

    @test_reference joinpath(REF_FLD, "acoustic2D_center_halo20.bson") Dict(:pend=>pend) by=comp
end

@testset "Test reference constant velocity freetop" begin
    # simple constant velocity model
    nx = ny = 201                           # grid size
    lx = ly = 2000.0                        # model sizes [m]
    vel_func(x,y) = 2000.0                  # velocity model [m/s]
    lt = 2.0                                # final time [s]
    # sources
    f0 = 10.0                               # source dominating frequency [Hz]
    t0 = 4 / f0                             # source activation time [s]
    stf = rickersource1D                    # second derivative of gaussian
    possrcs = zeros(1,2)
    possrcs[1,:] .= [lx/2, ly/2]
    srcs = Sources(possrcs, [t0], [stf], f0)
    # receivers
    posrecs = zeros(2,2)
    posrecs[1,:] .= [lx/2,  2ly/3]
    posrecs[2,:] .= [2lx/3, 2ly/3]
    recs = Receivers(posrecs)

    pend = solve2D_multi(lx, ly, lt, nx, ny, vel_func, srcs, recs;
            halo=20, rcoef=0.0001, do_vis=false, freetop=true, init_MPI=false)

    @test_reference joinpath(REF_FLD, "acoustic2D_center_freetop_halo20.bson") Dict(:pend=>pend) by=comp
end

MPI.Finalize()