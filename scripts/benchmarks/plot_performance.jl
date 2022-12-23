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

function plot_performance(result_files, out_file, title)
    results = []
    labels = []
    for (f, dim) in result_files
        # extract results
        res = readlines(f)
        sizes = []
        Teffs = []
        for l in res
            s, _, teff, _ = extract_result(l)
            push!(sizes, reduce(*, s))
            push!(Teffs, teff)
        end
        push!(results, (sizes, Teffs))
        push!(labels, split(basename(f), ".")[1])
    end

    # GPU memory bandwidth peak performance
    Tpeak_GPU = 559
    Tpeak_CPU = 22

    # plot reference lines
    maxylim = 2^10
    plot(
        2 .^ (9:2:29), fill(Tpeak_GPU, length(2 .^ (9:2:29)));
        markershape=:none, color=RGBA(0, 0, 0, 0.8),
        lw=3, linestyle=:dashdot,
        label="Tpeak GPU ($(round(Tpeak_GPU,sigdigits=3)) GB/s)"
    )
    plot!(
        2 .^ (9:2:29), fill(Tpeak_CPU, length(2 .^ (9:2:29)));
        markershape=:none, color=RGBA(0, 0, 0, 0.6),
        lw=3, linestyle=:dash,
        label="Tpeak CPU ($(round(Tpeak_CPU,sigdigits=3)) GB/s)"
    )
    
    for (r, lab) in zip(results, labels)
        plot!(
            r[1], r[2];
            xticks=2 .^ (10:2:28), yticks=(2 .^ (1:1:9), map(string, 2 .^ (1:1:9))),
            xlim=(2^(9), 2^(28.5)), ylim=(1.0,maxylim), xscale=:log2, yscale=:log2, label=lab, legend=:bottomright,
            xlabel="model size (number of grid points)", ylabel="Effective memory throughput [GB/s]", title=title
        )
    end

    # save plot as png
    png(out_file)
end

plot_performance([
    (joinpath(@__DIR__, "results", "acoustic_2D_CPU.txt"), 2),
    (joinpath(@__DIR__, "results", "acoustic_2D_GPU.txt"), 2),
    (joinpath(@__DIR__, "results", "acoustic_3D_CPU.txt"), 3),
    (joinpath(@__DIR__, "results", "acoustic_3D_GPU.txt"), 3)],
    joinpath(@__DIR__, "results", "performance_2D3D_perf.png"),
    "Effective performance of acoustic 2D/3D CPML on CPU and GPU"
)
