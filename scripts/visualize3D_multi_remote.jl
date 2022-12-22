import GLMakie
using Plots
using HDF5
using Printf

GLMakie.activate!()

REMOTE_PATH = "daint:~/scratch/AcousticWaveCPML.jl/docs/tmp"
LOCAL_PATH = joinpath(dirname(@__DIR__), "simulations", "tmp")

function download_frame(file_name, it, procs=8)
    mkpath(LOCAL_PATH)
    for p in 0:procs-1
        remote_file_path = joinpath(REMOTE_PATH, "$(file_name)_it$(it)_proc$(p).h5")
        local_file_path = joinpath(LOCAL_PATH, "$(file_name)_it$(it)_proc$(p).h5")
        run(`scp $(remote_file_path) $(local_file_path)`)
    end
end

function delete_local_frame(file_name, it, procs=8)
    for p in 0:procs-1
        local_file_path = joinpath(LOCAL_PATH, "$(file_name)_it$(it)_proc$(p).h5")
        run(`rm $(local_file_path)`)
    end
end

function load_frame(file_name, it, procs=8)
    # scp from remote
    download_frame(file_name, it)
    # load data
    lx = h5read(joinpath(LOCAL_PATH, "$(file_name)_it$(it)_proc0.h5"), "lx")
    ly = h5read(joinpath(LOCAL_PATH, "$(file_name)_it$(it)_proc0.h5"), "ly")
    lz = h5read(joinpath(LOCAL_PATH, "$(file_name)_it$(it)_proc0.h5"), "lz")
    halo = h5read(joinpath(LOCAL_PATH, "$(file_name)_it$(it)_proc0.h5"), "halo")
    possrcs = h5read(joinpath(LOCAL_PATH, "$(file_name)_it$(it)_proc0.h5"), "possrcs")
    # combine pressures
    if procs == 8
        pcur_p0 = h5read(joinpath(LOCAL_PATH, "$(file_name)_it$(it)_proc0.h5"), "pcur")
        pcur_p1 = h5read(joinpath(LOCAL_PATH, "$(file_name)_it$(it)_proc1.h5"), "pcur")
        pcur_p2 = h5read(joinpath(LOCAL_PATH, "$(file_name)_it$(it)_proc2.h5"), "pcur")
        pcur_p3 = h5read(joinpath(LOCAL_PATH, "$(file_name)_it$(it)_proc3.h5"), "pcur")
        pcur_p4 = h5read(joinpath(LOCAL_PATH, "$(file_name)_it$(it)_proc4.h5"), "pcur")
        pcur_p5 = h5read(joinpath(LOCAL_PATH, "$(file_name)_it$(it)_proc5.h5"), "pcur")
        pcur_p6 = h5read(joinpath(LOCAL_PATH, "$(file_name)_it$(it)_proc6.h5"), "pcur")
        pcur_p7 = h5read(joinpath(LOCAL_PATH, "$(file_name)_it$(it)_proc7.h5"), "pcur")
        # delete local frame files
        delete_local_frame(file_name, it)   

        nx, ny, nz = size(pcur_p0)
        gnx = (nx-2)*2 + 2
        gny = (ny-2)*2 + 2
        gnz = (nz-2)*2 + 2
        pcur = zeros(gnx, gny, gnz)

        pcur[1:nx    , 1:ny    , 1:nz    ] .= pcur_p0
        pcur[nx-1:gnx, 1:ny    , 1:nz    ] .= pcur_p4
        pcur[1:nx    , ny-1:gny, 1:nz    ] .= pcur_p2
        pcur[nx-1:gnx, ny-1:gny, 1:nz    ] .= pcur_p6
        pcur[1:nx    , 1:ny    , nz-1:gnz] .= pcur_p1
        pcur[nx-1:gnx, 1:ny    , nz-1:gnz] .= pcur_p5
        pcur[1:nx    , ny-1:gny, nz-1:gnz] .= pcur_p3
        pcur[nx-1:gnx, ny-1:gny, nz-1:gnz] .= pcur_p7
    end

    pcur .*= 1000
    return lx, ly, lz, halo, gnx, gny, gnz, pcur,possrcs
end

function visualise_3D_multi_remote(file_name; frames=[], threshold=0.0005, plims=(-1,1), fps=20)
    frame_names = []
    frameid = 1
    for frame in frames
        lx, ly, lz, halo, nx, ny, nz, pcur, possrcs = load_frame(file_name, frame)

        maxabsp = maximum(abs.(pcur))
        @show maxabsp

        pcur[(pcur .> plims[1] * threshold) .& (pcur .< plims[2] * threshold)] .= NaN

        pcur .= log10.(abs.(pcur))
        pcur = reverse(permutedims(pcur, [1, 3, 2]), dims=3)

        xc,yc,zc = LinRange(0,lx,nx),-reverse(LinRange(0,ly,ny)),LinRange(0,lz,nz)
        fig      = GLMakie.Figure(resolution=(1200,1000),fontsize=20,figure_padding=50)
        ax       = GLMakie.Axis3(fig[1,1]; aspect=:data,
                         title="3D multi-xPU Acoustic CPML\n(nx=$(nx), ny=$(ny), nz=$(nz), halo = $(halo), threshold=$(round(threshold * 100, digits=2))%)",
                         xlabel="lx",ylabel="lz",zlabel="ly")
        surf_T   = GLMakie.contour!(ax,xc,zc,yc,pcur;
            alpha=0.01,
            colormap=:diverging_bwr_20_95_c54_n256
        )

        dx, dy, dz = lx / (nx-1), ly / (ny-1), lz / (nz-1)
        GLMakie.scatter!(ax, possrcs[:,1] .* dx, possrcs[:,3] .* dz, .-possrcs[:,2] .* dy, markersize=20, marker=:star4, color="red", label="sources")
        
        push!(frame_names, @sprintf("%06d.png", frameid))
        GLMakie.save(joinpath(LOCAL_PATH,frame_names[frameid]),fig)
        frameid += 1
    end

    anim = Animation(LOCAL_PATH, frame_names)
    gif(anim, joinpath(dirname(@__DIR__), "simulations", "$(file_name).gif"); fps=fps)

    return nothing
end
