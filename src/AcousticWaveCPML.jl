module AcousticWaveCPML

    using ParallelStencil

    # Folders for saving documents
    DOCS_FLD = joinpath(dirname(@__DIR__), "simulations")
    TMP_FLD = joinpath(DOCS_FLD, "tmp")

    export Sources, Receivers, gaussource1D, rickersource1D

    # Utilites functions
    include("utils.jl")

    # 1D acoustic solver module
    include("Acoustic1D.jl")

    # 2D acoustic solver module
    include("Acoustic2D.jl")

    # 3D acoustic solver module
    include("Acoustic3D.jl")

    # xPU 2D acoustic solver module using Threads
    ParallelStencil.@reset_parallel_stencil()
    include("Acoustic2D_Threads.jl")

    # multixPU 2D acoustic solver module using Threads
    ParallelStencil.@reset_parallel_stencil()
    include("Acoustic2Dmulti_Threads.jl")

    # xPU 3D acoustic solver module using Threads
    ParallelStencil.@reset_parallel_stencil()
    include("Acoustic3D_Threads.jl")

    # multixPU 3D acoustic solver module using Threads
    ParallelStencil.@reset_parallel_stencil()
    include("Acoustic3Dmulti_Threads.jl")

    # xPU 3D acoustic solver module using CUDA
    ParallelStencil.@reset_parallel_stencil()
    include("Acoustic3D_CUDA.jl")

    # xPU 2D acoustic solver module using CUDA
    ParallelStencil.@reset_parallel_stencil()
    include("Acoustic2D_CUDA.jl")

    # multixPU 2D acoustic solver module using CUDA
    ParallelStencil.@reset_parallel_stencil()
    include("Acoustic2Dmulti_CUDA.jl")

    # multixPU 3D acoustic solver module using CUDA
    ParallelStencil.@reset_parallel_stencil()
    include("Acoustic3Dmulti_CUDA.jl")

end # module AcousticWaveCPML
