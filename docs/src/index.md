# AcousticWaveCPML.jl

_A Julia package for solving acoustic wave propagation in 2D and 3D on multi-xPUs using CPML boundary conditions._

## Package Features

- 1D, 2D and 3D acoustic wave equation solvers with CPML boundary conditions
- 2D and 3D **xPUs** (i.e. that can run on both CPUs and GPUs) implemenations (using [ParallelStencil.jl](https://github.com/omlins/ParallelStencil.jl))
- 2D and 3D **multi-xPUs** implementations (using [ParallelStencil.jl](https://github.com/omlins/ParallelStencil.jl) and [ImplicitGlobalGrid.jl](https://github.com/eth-cscs/ImplicitGlobalGrid.jl) as well as [MPI](https://github.com/JuliaParallel/MPI.jl))

## Usage

First of all, import the package with:
```julia
using AcousticWaveCPML
```

Then you have to import the submodule containing the specific version you want to run.
There are 1D, 2D and 3D submodules. For 2D and 3D there are versions with and without using `ParallelStencil`. For 2D and 3D solvers there are versions for multi-xPUs using `ParalleStencil` and `ImplicitGlobalGrid`.
Each submodule needs to be loaded separately and independently: this is because currently `ParallelStencil` does not easily support multiple initializations in the same module, so a workaround was needed.

Solvers that use `ParallelStencil` have either `_Threads` or `_CUDA` at the end of the name, respectively for CPU and GPU versions. Solvers that can be run on multi-xPUs have `multi` in their name.

For example, if you want to load the solver function for 2D acoustic wave propagation on GPUs, you will need to import with:
```julia
using AcousticWaveCPML.Acoustic2D_CUDA
```
and then use the function `solve2D` provided in the `Acoustic2D_CUDA` submodule.

See the [Index](@ref main-index) for the complete list of documented functions and types.

## GitHub repository

You will find most of the scientific info/results [here](https://github.com/GiackAloZ/AcousticWaveCPML.jl).

### [Index](@id main-index)

```@index
Pages = ["lib/public.md"]
```


