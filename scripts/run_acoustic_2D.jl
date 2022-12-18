include("solvers/acoustic_2D.jl")

# # gradient velocity model
# nx, ny = 211, 121
# vel = zeros(Float64, nx, ny);
# for i=1:nx
#     for j=1:ny
#         vel[i,j] = 2000.0 + 12.0*(j-1)
#     end
# end
# # constant after some depth
# bigv = vel[1,ny-40]
# vel[:,ny-40:end] .= bigv

# # 6 equidistant sources on top
# possrcs = zeros(Int,6,2)
# possrcs[1,:] = [div(3nx, 11, RoundUp), 3]
# possrcs[2,:] = [div(4nx, 11, RoundUp), 3]
# possrcs[3,:] = [div(5nx, 11, RoundUp), 3]
# possrcs[4,:] = [div(6nx, 11, RoundUp), 3]
# possrcs[5,:] = [div(7nx, 11, RoundUp), 3]
# possrcs[6,:] = [div(8nx, 11, RoundUp), 3]

# acoustic2D(2100.0, 1200.0, 1500, vel, possrcs; halo=5, rcoef=0.01, do_vis=true, gif_name="acoustic2D_halo5")
# acoustic2D(2100.0, 1200.0, 1500, vel, possrcs; halo=10, rcoef=0.001, do_vis=true, gif_name="acoustic2D_halo10")
# acoustic2D(2100.0, 1200.0, 1500, vel, possrcs; halo=20, rcoef=0.0001, do_vis=true, gif_name="acoustic2D_halo20")
# acoustic2D(2100.0, 1200.0, 1500, vel, possrcs; halo=40, rcoef=0.00001, do_vis=true, gif_name="acoustic2D_halo40")

# simple constant velocity model
nx, ny = 200, 200
vel = 2000.0 .* ones(Float64, nx, ny);
# one source in the center
possrcs = zeros(Int,1,2)
possrcs[1,:] = [div(nx, 2, RoundUp), div(ny, 2, RoundUp)]

# acoustic2D(2100.0, 2100.0, 1000, vel, possrcs; halo=5, rcoef=0.01, do_vis=true, gif_name="acoustic2D_center_halo5", freetop=false, threshold=0.001)
# acoustic2D(2100.0, 2100.0, 1000, vel, possrcs; halo=10, rcoef=0.001, do_vis=true, gif_name="acoustic2D_center_halo10", freetop=false, threshold=0.001)
acoustic2D(1990.0, 1990.0, 1000, vel, possrcs; halo=20, rcoef=0.0001, do_vis=true, gif_name="tmp1", freetop=false, threshold=0.001)
# acoustic2D(2100.0, 2100.0, 1000, vel, possrcs; halo=40, rcoef=0.00001, do_vis=true, gif_name="acoustic2D_center_halo40", freetop=false, threshold=0.001)
