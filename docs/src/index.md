# AcousticWaveCPML.jl

_A Julia package for solving acoustic wave propagation in 2D and 3D on multi-xPUs using CPML boundary conditions._

## Package Features

- 1D, 2D and 3D acoustic wave equation solvers with CPML boundary conditions
- 2D and 3D **xPUs** (i.e. that can run on both CPUs and GPUs) implemenations (using [ParallelStencil.jl](https://github.com/omlins/ParallelStencil.jl))
- 2D and 3D **multi-xPUs** implementations (using [ParallelStencil.jl](https://github.com/omlins/ParallelStencil.jl) and [ImplicitGlobalGrid.jl](https://github.com/eth-cscs/ImplicitGlobalGrid.jl) as well as [MPI](https://github.com/JuliaParallel/MPI.jl))

See the [Index](@ref main-index) for the complete list of documented functions and types.

## GitHub repository

You will find most of the scientific info/results [here](https://github.com/GiackAloZ/AcousticWaveCPML.jl).

### [Index](@id main-index)

```@index
Pages = ["lib/public.md"]
```


