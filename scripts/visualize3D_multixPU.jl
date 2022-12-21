using GLMakie
using Plots
using HDF5

GLMakie.activate!()

function load_frame(file_path, it, procs)
    if procs > 1
        if procs == 3
            pcur_p0 = h5read("$(file_path)_it$(it)_proc0.h5", "pcur")
            pcur_p1 = h5read("$(file_path)_it$(it)_proc1.h5", "pcur")
            pcur_p2 = h5read("$(file_path)_it$(it)_proc2.h5", "pcur")

            nx, ny, nz = size(pcur_p0)

            gnx = (nx-2)*3 + 2
            gny = ny
            gnz = nz

            pcur = zeros(gnx, gny, gnz)

            pcur[1:nx, :, :] .= pcur_p0
            pcur[nx-1:(nx-1)+nx-1, :, :] .= pcur_p1
            pcur[(nx-1)+nx-1-1:gnx, :, :] .= pcur_p2
        end
        if procs == 4
            pcur_p0 = h5read("$(file_path)_it$(it)_proc0.h5", "pcur")
            pcur_p1 = h5read("$(file_path)_it$(it)_proc1.h5", "pcur")
            pcur_p2 = h5read("$(file_path)_it$(it)_proc2.h5", "pcur")
            pcur_p3 = h5read("$(file_path)_it$(it)_proc3.h5", "pcur")

            nx, ny, nz = size(pcur_p0)

            gnx = (nx-2)*2 + 2
            gny = (ny-2)*2 + 2
            gnz = nz

            pcur = zeros(gnx, gny, gnz)

            pcur[1:nx, 1:ny, :] .= pcur_p0
            pcur[nx-1:gnx, 1:ny, :] .= pcur_p2
            pcur[1:nx, ny-1:gny, :] .= pcur_p1
            pcur[nx-1:gnx, ny-1:gny, :] .= pcur_p3
        end
    else
        pcur = h5read("$(file_path)_it$(it)_proc0.h5", "pcur")
        nx, ny, nz = size(pcur)
    end

    pcur .*= 1000

    return nx, ny, nz, pcur
end


function visualise_3D(file_name; it=1000, procs=1, frames=[], threshold=0.001, plims=(-3,3))
    file_path = joinpath(dirname(@__DIR__), "docs", "tmp", file_name)
    lx, ly, lz = h5read("$(file_path)_it$(it)_proc0.h5", "lx"), h5read("$(file_path)_it$(it)_proc0.h5", "ly"), h5read("$(file_path)_it$(it)_proc0.h5", "lz")
    halo = h5read("$(file_path)_it$(it)_proc0.h5", "halo")

    @show lx, ly, lz, halo

    frame_names = []
    if length(frames) > 0
        for frame in frames
            nx, ny, nz, pcur = load_frame(file_path, frame, procs)

            maxabsp = maximum(abs.(pcur))
            @show maxabsp

            pcur[(pcur .> plims[1] * threshold) .& (pcur .< plims[2] * threshold)] .= NaN

            pcur .= log10.(abs.(pcur))
            pcur = reverse(permutedims(pcur, [1, 3, 2]), dims=3)

            xc,yc,zc = LinRange(0,lx,nx),-reverse(LinRange(0,ly,ny)),LinRange(0,lz,nz)
            fig      = Figure(resolution=(1600,1000),fontsize=24,figure_padding=100)
            ax       = Axis3(fig[1,1]; aspect=(1,1,1),
                             title="3D Acoustic CPML multi-xPU\n(halo = $(halo), threshold=$(round(threshold * 100, digits=2))%)",
                             xlabel="lx",ylabel="lz",zlabel="ly")
            surf_T   = GLMakie.contour!(ax,xc,zc,yc,pcur;
                alpha=0.01,
                colormap=:diverging_bwr_20_95_c54_n256
            )
            GLMakie.zlims!(-ly,0)

            # GLMakie.scatter!(ax,[lx/3, lx/3, 2lx/3, 2lx/3],[lz/3, 2lz/3, lz/3, 2lz/3],[3, 3, 3, 3], markersize=20, marker=:star4, color="red", label="sources")
            
            push!(frame_names, "$(file_name)_it$(frame).png")
            save(joinpath(dirname(@__DIR__),"docs","tmp","$(file_name)_it$(frame).png"),fig)
        end
    end

    return nothing
end
