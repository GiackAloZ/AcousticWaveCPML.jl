push!(LOAD_PATH, "../src")

using AcousticWaveCPML

function runtests()
    exename = joinpath(Sys.BINDIR, Base.julia_exename())
    testdir = pwd()

    printstyled("Testing AcousticWaveCPML.jl\n"; bold=true, color=:white)

    printstyled("Testing 1D AcousticWaveCPML\n"; bold=true, color=:white)
    run(`$exename -O3 --startup-file=no --check-bounds=yes $(joinpath(testdir, "test1D.jl"))`)
    printstyled("Testing 2D AcousticWaveCPML\n"; bold=true, color=:white)
    run(`$exename -O3 --startup-file=no --check-bounds=yes $(joinpath(testdir, "test2D.jl"))`)
    printstyled("Testing 2D xPU AcousticWaveCPML\n"; bold=true, color=:white)
    run(`$exename -O3 --startup-file=no --check-bounds=yes $(joinpath(testdir, "test2D_xPU.jl"))`)
    printstyled("Testing 2D multixPU AcousticWaveCPML\n"; bold=true, color=:white)
    run(`$exename -O3 --startup-file=no --check-bounds=yes $(joinpath(testdir, "test2D_multixPU.jl"))`)
    printstyled("Testing 3D AcousticWaveCPML\n"; bold=true, color=:white)
    run(`$exename -O3 --startup-file=no --check-bounds=yes $(joinpath(testdir, "test3D.jl"))`)
    printstyled("Testing 3D xPU AcousticWaveCPML\n"; bold=true, color=:white)
    run(`$exename -O3 --startup-file=no --check-bounds=yes $(joinpath(testdir, "test3D_xPU.jl"))`)
    printstyled("Testing 3D multixPU AcousticWaveCPML\n"; bold=true, color=:white)
    run(`$exename -O3 --startup-file=no --check-bounds=yes $(joinpath(testdir, "test3D_multixPU.jl"))`)

    return 0
end

exit(runtests())
