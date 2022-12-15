include("solvers/acoustic_1D.jl")

# simple constant velocity model
vel = 2000 .* ones(Float64, 201)

acoustic1D(2000.0, 500, vel; halo=5, rcoef=0.01, do_vis=true, gif_name="acoustic1D_halo5")
acoustic1D(2000.0, 500, vel; halo=10, rcoef=0.001, do_vis=true, gif_name="acoustic1D_halo10")
acoustic1D(2000.0, 500, vel; halo=20, rcoef=0.0001, do_vis=true, gif_name="acoustic1D_halo20")
