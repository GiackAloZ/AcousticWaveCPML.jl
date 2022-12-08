@views function update_psi!(psi_left, psi_right, pcur, halo, nx,
                            K_x_half_left, K_x_half_right,
                            a_x_half_left, a_x_half_right,
                            b_x_half_left, b_x_half_right)
    for i = 2:halo
        ii = i + nx + halo - 1
        psi_left[i]  =  b_x_half_left[i] /  K_x_half_left[i] * psi_left[i] +  a_x_half_left[i] * (pcur[i +1] - pcur[i ])
        psi_right[i] = b_x_half_right[i] / K_x_half_right[i] * psi_left[i] + a_x_half_right[i] * (pcur[ii+1] - pcur[ii])
    end
end

@views function update_p_CPML!(xi_left, xi_right, pold, pcur, pnew, vel2dt2_dx2)
    for ii=(1,3)
        for j = 2:nz-1 # 2:nz-1 !!
                for i = cpml.ipmlidxs[ii]:cpml.ipmlidxs[ii+1]
                
                d2pdx2 = pcur[i+1,j]-2.0*pcur[i,j]+pcur[i-1,j]            
                d2pdz2 = pcur[i,j+1]-2.0*pcur[i,j]+pcur[i,j-1]
                dpsidx = psi_x[i,j] - psi_x[i-1,j] 
                dpsidz = psi_z[i,j] - psi_z[i,j-1] 
                
                xi_x[i,j] = cpml.b_x[i] / cpml.K_x_half[i] * xi_x[i,j] + cpml.a_x[i] * (d2pdx2 + dpsidx)
                xi_z[i,j] = cpml.b_z[j] / cpml.K_z_half[j] * xi_z[i,j] + cpml.a_z[j] * (d2pdz2 + dpsidz)
                            
                damp = fact[i,j] * (dpsidx + dpsidz + xi_x[i,j] + xi_z[i,j])

                # update pressure
                pnew[i,j] = 2.0*pcur[i,j] -pold[i,j] + fact[i,j]*(d2pdx2 + d2pdz2) + damp
            end
        end
    end
end

@views function update_p!()
    for j = cpml.jpmlidxs[2]+1:cpml.jpmlidxs[3]-1    #2:nz-1 
        for i = cpml.ipmlidxs[2]+1:cpml.ipmlidxs[3]-1   #2:nx-1 

          d2pdx2 = pcur[i+1,j]-2.0*pcur[i,j]+pcur[i-1,j]            
          d2pdz2 = pcur[i,j+1]-2.0*pcur[i,j]+pcur[i,j-1]
          
          pnew[i,j] = 2.0*pcur[i,j] -pold[i,j] + fact[i,j]*(d2pdx2 + d2pdz2)
      end
    end
end

@views function inject_sources!()
    
end

@views function acoustic1D(
    nx::Integer,
    vel::Vector{Float64},
    nt::Integer = 1500,
    halo::Integer = 21
)
    # Physics
    lx = 100.0                          # length of model [m]
    f0 = 12.0                           # dominating frequency [Hz]
    # Derived physics
    vel_max = maximum(vel)              # maximum velocity [m/s]
    # Numerics
    npower        = 2.0
    K_max         = 1.0
    rcoef         = 0.0001
    # Derived numerics
    dx = lx / nx                        # size of grid cell [m]
    xc = LinRange(-lx/2,lx/2,nx)        # cell positions [m]
    dt = dx / vel_max                   # timestep size (CFL + Courant condition) [s]
    # CPML numerics
    alpha_max        = 2.0*π*(f0/2.0)
    thickness_cpml_x = halo * dx
    d0_x             = -(npower + 1) * vel_max * log(rcoef) / (2.0 * thickness_cpml_x)
    K_x_left,K_x_right,a_x_left,a_x_right,b_x_left,b_x_right   = calc_Kab_CPML(nx_tot,halo,dx,dt,npower,d0_x,alpha_max,K_max,"ongrd")
    K_x_half_left,K_x_half_left_right,a_x_half_left,a_x_half_right,b_x_half_left,b_x_half_right = calc_Kab_CPML(nx_tot,halo,dx,dt,npower,d0_x,alpha_max,K_max,"halfgrd")
    
    # Array initialization

    # pressure arrays
    pold = zeros(nx)
    pcur = zeros(nx)
    pnew = zeros(nx)
    # CPML arrays
    psi_left, psi_right = zeros(halo+1), zeros(halo+1)
    xi_left, xi_right = zeros(halo), zeros(halo)

    
end