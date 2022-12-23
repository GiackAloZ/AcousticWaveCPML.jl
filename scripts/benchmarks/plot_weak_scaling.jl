using Plots, Plots.Measures
pyplot()

default(
    size=(1600,1000),
    framestyle=:box,
    margin=15mm,
    lw=5,
    minorgrid=true,
    grid=true,
    markershape=:circle,
    markersize=10,
    labelfontsize=20,
    legendfontsize=14,
    tickfontsize=20,
    titlefontsize=24
)

include("utils.jl")

function plot_weak_scaling(out_file, title)
    baseline_2D_single_xPU = 1.43448780e+01
    baseline_3D_single_xPU = 3.00937692e+00

    nprocs_2D = [1, 4, 9, 16, 25, 36]
    nprocs_3D = [1, 8, 27, 64]

    times_multi2D = [
        1.43448780e+01,
        1.49460790e+01,
        1.49583530e+01,
        1.49448981e+01,
        1.49450178e+01,
        1.49515481e+01
    ]
    times_multi3D = [
        3.00937692e+00,
        3.02247500e+00,
        3.02497411e+00,
        3.02258301e+00
    ]

    weak_eff_multi2D = baseline_2D_single_xPU ./ times_multi2D .* 100
    weak_eff_multi3D = baseline_3D_single_xPU ./ times_multi3D .* 100
    
    plot(
        nprocs_2D, weak_eff_multi2D;
        xticks=2 .^ (0:8), ylim=(90,101), xscale=:log2, label="acoustic_2D_multixPU", legend=:bottomright,
        xlabel="number of nodes/GPUs", ylabel="Weak scaling efficiency [%]", title=title
    )
    plot!(
        nprocs_3D, weak_eff_multi3D;
        label="acoustic_3D_multixPU"
    )

    # save plot as png
    png(out_file)
end

plot_weak_scaling(
    joinpath(@__DIR__, "results", "weak_scaling_eff_2D3D.png"),
    "Weak scaling efficiency of acoustic 2D/3D CPML on multiple GPUs"
)
