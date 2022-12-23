import GLMakie
using Plots
using HDF5

GLMakie.activate!()

"""
    load_frame(file_path, it)

Load the frame of iteration `it` from `file_path`
"""
function load_frame(file_path, it)
    lx, ly, lz = h5read("$(file_path)_it$(it).h5", "lx"), h5read("$(file_path)_it$(it).h5", "ly"), h5read("$(file_path)_it$(it).h5", "lz")
    halo = h5read("$(file_path)_it$(it).h5", "halo")
    possrcs = h5read("$(file_path)_it$(it).h5", "possrcs")
    pcur = h5read("$(file_path)_it$(it).h5", "pcur")
    pcur .*= 1000

    nx, ny, nz = size(pcur)

    return lx, ly, lz, halo, possrcs, nx, ny, nz, pcur
end

"""
    visualise_3D(file_name; it=1000, frames=[], threshold=0.001, plims=(-3,3), fps=20)

Visualise 3D animation from tmp folder in simulations with prefix `file_name`, frames to pick for animation `frames`,
limits of pressure for visualisation `plims`, percentage of `plims` pressure values to cut `threshold`, gif fps `fps`.

Save the resulting animation in the simulations folder with name `file_name.gif`.
"""
function visualise_3D(file_name; frames=[], threshold=0.001, plims=(-3,3), fps=20)
    file_path = joinpath(dirname(@__DIR__), "simulations", "tmp", file_name)

    frame_names = []
    frameid = 1
    for frame in frames
        lx, ly, lz, halo, possrcs, nx, ny, nz, pcur = load_frame(file_path, frame)

        maxabsp = maximum(abs.(pcur))
        @show maxabsp

        pcur[(pcur .> plims[1] * threshold) .& (pcur .< plims[2] * threshold)] .= NaN

        pcur .= log10.(abs.(pcur))
        pcur = reverse(permutedims(pcur, [1, 3, 2]), dims=3)

        xc,yc,zc = LinRange(0,lx,nx),-reverse(LinRange(0,ly,ny)),LinRange(0,lz,nz)
        fig      = GLMakie.Figure(resolution=(1600,1000),fontsize=24,figure_padding=100)
        ax       = GLMakie.Axis3(fig[1,1]; aspect=:data, viewmode=:fit,
                         title="3D xPU Acoustic CPML\n(nx=$(nx), ny=$(ny), nz=$(nz), halo = $(halo), threshold=$(round(threshold * 100, digits=2))%)",
                         xlabel="lx",ylabel="lz",zlabel="ly")
        surf_T   = GLMakie.contour!(ax,xc,zc,yc,pcur;
            alpha=0.01,
            colormap=:diverging_bwr_20_95_c54_n256
        )
        GLMakie.zlims!(-ly,0)

        dx, dy, dz = lx / (nx-1), ly / (ny-1), lz / (nz-1)
        GLMakie.scatter!(ax, possrcs[:,1] .* dx, possrcs[:,3] .* dz, .-possrcs[:,2] .* dy, markersize=20, marker=:star4, color="red", label="sources")
        
        push!(frame_names, @sprintf("%06d.png", frameid))
        GLMakie.save(joinpath(dirname(@__DIR__),"simulations","tmp",frame_names[frameid]),fig)
        frameid += 1
    end

    anim = Animation(joinpath(dirname(@__DIR__),"simulations","tmp"), frame_names)
    gif(anim, joinpath(dirname(@__DIR__), "simulations", "$(file_name).gif"); fps=fps)

    return nothing
end
