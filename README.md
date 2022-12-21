# AcousticWaveCPML.jl

[![Build Status](https://github.com/GiackAloZ/AcousticWaveCPML.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/GiackAloZ/AcousticWaveCPML.jl/actions/workflows/CI.yml)

A Julia package for solving acoustic wave propagation in 2D and 3D on multi-xPUs using CPML boundary conditions.

This package also provide functionalities to compute gradients with respect to velocities using the adjoint method and residuals from available data. Checkpointing features also allow to compute gradients without running out of memory.
