using Test
using DSP, NumericalIntegration, LinearAlgebra

using AcousticWaveCPML
using AcousticWaveCPML.Acoustic1D

@testset "Test homogeneous velocity analytical solution" begin
    # simple constant velocity model
    nx = 1001                               # grid size
    lx = 10.0                                # model sizes [m]
    c0 = 1.0
    vel = c0 .* ones(Float64, nx)           # velocity model [m/s]
    lt = 10.0                                # final time [s]
    # sources
    f0 = 1.0                                # source dominating frequency [Hz]
    t0 = 4 / f0                             # source activation time [s]
    stf = rickersource1D                    # second derivative of gaussian
    possrcs = zeros(1,1)
    possrcs[1,:] .= [lx/2]
    srcs = Sources(possrcs, [t0], [stf], f0)
    # receivers
    posrecs = zeros(1,1)
    posrecs[1,:] .= [lx/4]
    recs = Receivers(posrecs)

    # numberical solution
    solve1D(lx, lt, vel, srcs, recs; halo=0)
    numerical_trace = recs.seismograms[:,1]

    # Analytical solution
    dx = lx / (nx-1)
    dt = dx / c0
    nt = ceil(Int, lt / dt)
    times = collect(range(0.0, step=dt, length=nt))
    dist = norm(possrcs[1,:] .- posrecs[1,:])
    src = rickersource1D.(times, t0, f0)
    # Calculate Green's function
    G = times .* 0.
    for it = 1:nt
        # Heaviside function
        if (times[it] - dist / c0) >= 0
            G[it] = 1. / (2 * c0)
        end
    end
    # Convolve with source term
    Gc = conv(G, src .* dt)
    Gc = Gc[1:nt]

    @test length(numerical_trace) == length(Gc) == nt
    # test integral of absolute difference over time is less then a constant 1% error relative to the peak analytical solution
    @test integrate(times, abs.(numerical_trace .- Gc)) <= maximum(abs.(Gc)) * 0.01 * lt
end

@testset "Test homogeneous velocity analytical solution CPML halo 20" begin
    # simple constant velocity model
    nx = 1001                               # grid size
    lx = 10.0                                # model sizes [m]
    c0 = 1.0
    vel = c0 .* ones(Float64, nx)           # velocity model [m/s]
    lt = 20.0                                # final time [s]
    # sources
    f0 = 1.0                                # source dominating frequency [Hz]
    t0 = 4 / f0                             # source activation time [s]
    stf = rickersource1D                    # second derivative of gaussian
    possrcs = zeros(1,1)
    possrcs[1,:] .= [lx/2]
    srcs = Sources(possrcs, [t0], [stf], f0)
    # receivers
    posrecs = zeros(1,1)
    posrecs[1,:] .= [lx/4]
    recs = Receivers(posrecs)

    # numberical solution
    solve1D(lx, lt, vel, srcs, recs; halo=20)
    numerical_trace = recs.seismograms[:,1]

    # Analytical solution
    dx = lx / (nx-1)
    dt = dx / c0
    nt = ceil(Int, lt / dt)
    times = collect(range(0.0, step=dt, length=nt))
    dist = norm(possrcs[1,:] .- posrecs[1,:])
    src = rickersource1D.(times, t0, f0)
    # Calculate Green's function
    G = times .* 0.
    for it = 1:nt
        # Heaviside function
        if (times[it] - dist / c0) >= 0
            G[it] = 1. / (2 * c0)
        end
    end
    # Convolve with source term
    Gc = conv(G, src .* dt)
    Gc = Gc[1:nt]

    @test length(numerical_trace) == length(Gc) == nt
    # test integral of absolute difference over time is less then a constant 1% error relative to the peak analytical solution
    @test integrate(times, abs.(numerical_trace .- Gc)) <= maximum(abs.(Gc)) * 0.01 * lt
end