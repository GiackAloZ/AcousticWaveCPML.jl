

using HDF5
using Interpolations


"""
    rescalemod(nx,ny,nz;
               kind="nearest",
               flnamedset=[joinpath(@__DIR__, "foldsfaultmod3D.h5"), "foldsfaultmod"],
               func=false)

Rescale/interpolate a model given as a 3D array stored in a HDF5 file.

nx, ny and nz are the new sizes along the three axes.

Interpolation is done with BSplines, "kind" determines the order.

If func is true, then it return a callable "function-like" object instead.
"""
function rescalemod(nx,ny,nz;
                    kind="nearest",
                    flnamedset=[joinpath(@__DIR__, "foldsfaultmod3D.h5"),"foldsfaultmod"],
                    func=false)
  
    res = h5read(flnamedset[1],flnamedset[2])

    nxorig,nyorig,nzorig = size(res)

    x = LinRange(1.0, nxorig, nx)
    y = LinRange(1.0, nyorig, ny)
    z = LinRange(1.0, nzorig, nz)

    if kind=="nearest"
        ## nearest neighbor
        itp = interpolate(res,BSpline(Constant()))
    elseif kind=="linear"
        ## linear
        itp = interpolate(res,BSpline(Linear()))
    elseif kind=="quadratic" 
        ## quadratic
        itp = interpolate(res,BSpline(Quadratic(Natural(OnCell()))))
    elseif kind=="cubic" 
        ## cubic
        itp = interpolate(res,BSpline(Cubic(Natural(OnCell()))))
    end

    if func
        return itp
    end

    res_itp = itp(x,y,z)    

    return res_itp
end
