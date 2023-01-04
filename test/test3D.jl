using Test, ReferenceTests, BSON
using DSP, NumericalIntegration, LinearAlgebra

using AcousticWaveCPML
using AcousticWaveCPML.Acoustic3D
import AcousticWaveCPML.Acoustic3D: update_ψ!

REF_FLD = joinpath(@__DIR__, "references")

@testset "Test update_ψ!" begin
    nx = ny = nz = 101
    halo = 20
    pcur = ones(nx, ny, nz)
    _dx = _dy = _dz = 1 / 10.0
    pcur[[1,end],:,:] .= 0.0
    pcur[:,[1,end],:] .= 0.0
    pcur[:,:,[1,end]] .= 0.0

    ψ_x_l, ψ_x_r, ψ_y_l, ψ_y_r, ψ_z_l, ψ_z_r = ones(halo+1, ny, nz), ones(halo+1, ny, nz), ones(nx, halo+1, nz), ones(nx, halo+1, nz), ones(nx, ny, halo+1), ones(nx, ny, halo+1)
    a_x_hl, a_x_hr = 0.5 .* ones(halo+1), 2 .* ones(halo+1)
    b_K_x_hl, b_K_x_hr = 0.5 .* ones(halo+1), 2 .* ones(halo+1)
    a_y_hl, a_y_hr = 0.5 .* ones(halo+1), 2 .* ones(halo+1)
    b_K_y_hl, b_K_y_hr = 0.5 .* ones(halo+1), 2 .* ones(halo+1)
    a_z_hl, a_z_hr = 0.5 .* ones(halo+1), 2 .* ones(halo+1)
    b_K_z_hl, b_K_z_hr = 0.5 .* ones(halo+1), 2 .* ones(halo+1)

    update_ψ!(ψ_x_l, ψ_x_r, ψ_y_l, ψ_y_r, ψ_z_l, ψ_z_r, pcur,
              halo, _dx, _dy, _dz,
              a_x_hl, a_x_hr,
              b_K_x_hl, b_K_x_hr,
              a_y_hl, a_y_hr,
              b_K_y_hl, b_K_y_hr,
              a_z_hl, a_z_hr,
              b_K_z_hl, b_K_z_hr)
    
    a1 = 0.55 .* ones(ny,nz)
    a1[[1,end],:] .= 0.5
    a1[:,[1,end]] .= 0.5
    @test ψ_x_l[1,:,:] == a1
    @test ψ_x_l[2:end,:,:] == 0.5 .* ones(halo, ny, nz)
    a2 = 1.8 .* ones(ny,nz)
    a2[[1,end],:] .= 2.0
    a2[:,[1,end]] .= 2.0
    @test ψ_x_r[1:end-1,:,:] == 2.0 .* ones(halo, ny, nz)
    @test ψ_x_r[end,:,:] == a2

    a3 = 0.55 .* ones(nx,nz)
    a3[[1,end],:] .= 0.5
    a3[:,[1,end]] .= 0.5
    @test ψ_y_l[:,1,:] == a3
    @test ψ_y_l[:,2:end,:] == 0.5 .* ones(nx, halo, nz)
    a4 = 1.8 .* ones(nx,nz)
    a4[[1,end],:] .= 2.0
    a4[:,[1,end]] .= 2.0
    @test ψ_y_r[:,1:end-1,:] == 2.0 .* ones(nx, halo, nz)
    @test ψ_y_r[:,end,:] == a2

    a5 = 0.55 .* ones(ny,nz)
    a5[[1,end],:] .= 0.5
    a5[:,[1,end]] .= 0.5
    @test ψ_z_l[:,:,1] == a5
    @test ψ_z_l[:,:,2:end] == 0.5 .* ones(nx, ny, halo)
    a6 = 1.8 .* ones(nz,nz)
    a6[[1,end],:] .= 2.0
    a6[:,[1,end]] .= 2.0
    @test ψ_z_r[:,:,1:end-1] == 2.0 .* ones(nx, ny, halo)
    @test ψ_z_r[:,:,end] == a6
end

@testset "Test homogeneous velocity analytical solution" begin
    # simple constant velocity model
    nx = ny = nz = 101                      # grid size
    lx = ly = lz = 20.0                     # model sizes [m]
    c0 = 1.0
    vel = c0 .* ones(Float64, nx, ny, nz)   # velocity model [m/s]
    lt = 26.0                               # final time [s]
    # sources
    f0 = 0.25                                # source dominating frequency [Hz]
    t0 = 4 / f0                             # source activation time [s]
    stf = rickersource1D                    # second derivative of gaussian
    possrcs = zeros(1,3)
    possrcs[1,:] .= [lx/2, ly/2, lz/2]
    srcs = Sources(possrcs, [t0], [stf], f0)
    # receivers
    posrecs = zeros(1,3)
    posrecs[1,:] .= [lx/4, ly/2, lz/2]
    recs = Receivers(posrecs)

    # Numerical solution
    solve3D(lx, ly, lz, lt, vel, srcs, recs; halo=0, freetop=false)
    numerical_trace = recs.seismograms[:,1]

    # Analytical solution
    dx = lx / (nx-1)
    dy = ly / (ny-1)
    dz = lz / (nz-1)
    dt = sqrt(3) / (c0 * (1/dx + 1/dy + 1/dz))
    nt = ceil(Int, lt / dt)
    times = collect(range(0.0, step=dt, length=nt+1))
    dist = norm(possrcs[1,:] .- posrecs[1,:])
    src = rickersource1D.(times, t0, f0)
    # Calculate Green's function
    G = times .* 0.
    for it = 1:nt+1
        # Delta function
        if (times[it] - dist / c0) >= 0
            G[it] = 1.0 / (4π * c0^2 * dist)
            break
        end
    end
    Gc = conv(G, src)
    Gc = Gc[2:nt+1]

    @test length(numerical_trace) == length(Gc) == nt
    # test integral of absolute difference over time is less then a constant 1% error relative to the peak analytical solution
    @test integrate(times[2:end], abs.(numerical_trace .- Gc)) <= maximum(abs.(Gc)) * 0.01 * lt
end

@testset "Test homogeneous velocity analytical solution CPML halo 20" begin
    # simple constant velocity model
    nx = ny = nz = 101                      # grid size
    lx = ly = lz = 20.0                     # model sizes [m]
    c0 = 1.0
    vel = c0 .* ones(Float64, nx, ny, nz)   # velocity model [m/s]
    lt = 50.0                               # final time [s]
    # sources
    f0 = 0.25                                # source dominating frequency [Hz]
    t0 = 4 / f0                             # source activation time [s]
    stf = rickersource1D                    # second derivative of gaussian
    possrcs = zeros(1,3)
    possrcs[1,:] .= [lx/2, ly/2, lz/2]
    srcs = Sources(possrcs, [t0], [stf], f0)
    # receivers
    posrecs = zeros(1,3)
    posrecs[1,:] .= [lx/4, ly/2, lz/2]
    recs = Receivers(posrecs)

    # Numerical solution
    solve3D(lx, ly, lz, lt, vel, srcs, recs; halo=20, freetop=false)
    numerical_trace = recs.seismograms[:,1]

    # Analytical solution
    dx = lx / (nx-1)
    dy = ly / (ny-1)
    dz = lz / (nz-1)
    dt = sqrt(3) / (c0 * (1/dx + 1/dy + 1/dz))
    nt = ceil(Int, lt / dt)
    times = collect(range(0.0, step=dt, length=nt+1))
    dist = norm(possrcs[1,:] .- posrecs[1,:])
    src = rickersource1D.(times, t0, f0)
    # Calculate Green's function
    G = times .* 0.
    for it = 1:nt+1
        # Delta function
        if (times[it] - dist / c0) >= 0
            G[it] = 1.0 / (4π * c0^2 * dist)
            break
        end
    end
    Gc = conv(G, src)
    Gc = Gc[2:nt+1]

    @test length(numerical_trace) == length(Gc) == nt
    # test integral of absolute difference over time is less then a constant 1% error relative to the peak analytical solution
    @test integrate(times[2:end], abs.(numerical_trace .- Gc)) <= maximum(abs.(Gc)) * 0.01 * lt
end
