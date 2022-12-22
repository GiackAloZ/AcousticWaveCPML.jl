# AcousticWaveCPML.jl

[![Build Status](https://github.com/GiackAloZ/AcousticWaveCPML.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/GiackAloZ/AcousticWaveCPML.jl/actions/workflows/CI.yml)

A Julia package for solving acoustic wave propagation in 2D and 3D on multi-xPUs using CPML boundary conditions.

This package will also provide functionalities to compute gradients with respect to velocities using the adjoint method and residuals from available data. Checkpointing features also allow to compute gradients without running out of memory.

## Introduction and motivation

Seismic tomography is the predominant methodology geoscientists use to investigate the structure and properties
of the inaccessible Earth’s interior. Inferring properties of the subsurface from seismic data measured on
the surface of the Earth, i.e., solving the tomographic inverse problem, requires the simulation of the physics of
wave propagation (the forward problem).

The forward problem can be solved numerically by suitable methods
for PDEs (boundary value problems). These methods quickly become computationally expensive when a fine
discretization of the subsurface is needed and/or when working with 3D models. Because of that, exploiting
parallelization and computational resources is crucial to obtain results in a reasonable amount of time, allowing
for the solution of the inverse problem.

In particular, GPUs are becoming increasingly employed to speed-up
computations of finite difference methods, because of the regularity of their computation pattern.

Currently, there is a lack of availability of open source software packages combining easiness of use, good
performance and scalability to address the above-mentioned calculations. The Julia programming language is
then a natural candidate to fill such gap. Julia has recently emerged among computational physicists because
of a combination of easiness of use (reminiscent of languages like Python and MatLab) and its speed. Moreover,
Julia natively supports several parallelization paradigms (such as shared memory, message passing and GPU
computing) and recently developed packages like ParallelStencil.jl, which make for a simple, yet powerful, way
to implement scalable, efficient, maintainable and hardware-agnostic parallel algorithms.

We have implemented a time-domain finite-difference solvers for acoustic wave propagation
with C-PML boundary conditions on GPUs using Julia. Extensive benchmarks and performance evaluations of the developed parallel code have been conducted to assess the gain in performance and the possbility to scale on multi-GPUs.

## Physical model

TODO

## Numerical methods and implementation

TODO

## Model setups

TODO

## Results

TODO

### 1D CPML

### 2D CPML

### 3D CPML

## Conclusions

TODO
