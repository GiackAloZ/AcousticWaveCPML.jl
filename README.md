# AcousticWaveCPML.jl

[![Build Status](https://github.com/GiackAloZ/AcousticWaveCPML.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/GiackAloZ/AcousticWaveCPML.jl/actions/workflows/CI.yml)

A Julia package for solving acoustic wave propagation in 2D and 3D on multi-xPUs using CPML boundary conditions.

_WIP: this package will also provide functionalities to compute gradients with respect to velocities using the adjoint method and residuals from available data. Checkpointing features will also allow to compute gradients without running out of memory._

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

We have implemented some time-domain finite-difference solvers for acoustic wave propagation
with CPML boundary conditions on GPUs using Julia. Extensive benchmarks and performance evaluations of the developed parallel code have been conducted to assess the gain in performance and the possbility to scale on multi-GPUs.

## Physical model

The main PDE to solve is the classical acoustic wave equation:

$$
\frac{\partial^2 p}{\partial t^2} = c^2 \nabla^2 p,
$$
where $p$ is the pressure and $c$ is the wave propagation velocity (i.e. speed of sound). We consider from now on the pressure term $p$ to be the difference from lithostatic pressure. Hence we consider as initial values for pressure $p(x,0) = 0$.

If we also consider a source term $s(x,t)$ the function becomes:

$$
\frac{\partial^2 p}{\partial t^2} = c^2 \nabla^2 p + s
$$


Consider some rectangular area of space $\Omega \in \mathbb{R}^n, n = 1, 2, 3$. Consider also a fine time $T$ at which we end our simulation. Then we can choose an appropriate boundary condition on the boundary $\partial \Omega$ of $\Omega$ and we have a boundary value problem we can solve on the space-time set $\Omega \times [0,T]$.

If we choose a free boundary condition (i.e. an homogenous Dirichlet BDC), we get the following BVP:

$$
\begin{align*}
\frac{\partial^2 p}{\partial t^2} &= c^2 \nabla^2 p + s &&,~ \text{in}~ \Omega &&\times [0,T] \\
p &= 0 &&,~ \text{in}~ \partial \Omega &&\times [0,T] \\
\end{align*}
$$

### CPML boundary conditions

The main issue with this type of BDC is that waves are reflected at the boundaries. If we want to use our solver as forward and adjoint solver for the inverse problem, this causes spurious reflections at the boundaries that we did not have during the experiments in which we gathered date from receivers. To solve this problem, we have adopted the so-called CPML boundary conditions, which are used as a kind of absorbing BDCs that simulate a much bigger area of space then our area of interest $\Omega$.

The main concept of CPML BDCs is to enhance the standard wave equation at the boundaries with additional terms $\psi_i$ and $\xi_i$ that are used to damp the waves:

$$
\frac{\partial^2 p}{\partial t^2} = c^2 \left( \nabla^2 p + \left( \frac{\partial \psi_i}{\partial i} \right) + \xi_i \right),~ \text{in}~ \tilde{\partial}_i \Omega \times [0,T]
$$
where $i$ is used as Einstein notation to denote the various dimensions (e.g. $i \in \{x,y,z\}$ in 3D) and $\tilde{\partial}_i \Omega$ denotes an extension of $\partial \Omega$ in the $i$ dimension to create a thicker boundary layer.

These two new fields $\psi_i$ and $\xi_i$ evolution in time is expressed by the following PDEs:

$$
\begin{align}
\psi^n_i &= \frac{b_i}{K_i} \psi^{n-1}_i &&+ a_i \left( \frac{\partial p}{\partial i} \right)^n, \\
\xi^n_i &= \frac{b_i}{K_i} \xi^{n-1}_i &&+ a_i \left[ \left( \frac{\partial^2 p}{\partial i^2} \right)^n + \left( \frac{\partial \psi}{\partial i} \right)^n \right],
\end{align}
$$
where $n$ is an index that represents the current time step, not a power exponent. Initialization of the fields is straight-forward:
$$
\psi^0_i = 0, \xi^0_i = 0, \forall i~ \text{in}~ \tilde{\partial} \Omega
$$

The coefficients $a_i, b_i, K_i$ are computed in the following way:

$$
\begin{align*}
K_i(x) &= 1, \\
b_i(x) &= \exp(- \Delta t (D_i(x) + \alpha_i(x))), \\
a_i(x) &= D_i(x) \frac{b_i(x) - 1.0}{K_i(x) (D_i(x) + K_i(x) \alpha_i(x))}
\end{align*}
$$
and
$$
\begin{align*}
D_i(x) &= \frac{-(N + 1) \max(c) \log(R)}{2t_i} {d_i(x)}^{N}, \\
\alpha_i(x) &= \pi f (1 - d_i(x)),
\end{align*}
$$
where:
- $t_i$ is the thickness (in meters) of the CPML boundary in dimension $i$,
- $N$ is a power coefficient (we used $N=2$),
- $f$ is the dominating frequency of the source $s$,
- $R$ is the reflection coefficient (different values where used depending on the thickness of the CPML boundary),
- $d_i(x)$ is the normalized distance (between $0$ and $1$) from $x$ to the interior part $\overline{\Omega}$ of $\Omega$ in dimension $i$.

We picked these experimentally determined coefficients from reference studies of CPML BDCs on acoustic wave equation.

One crucial observation is that each component of the $\psi_i$ and $\xi_i$ fields is non-zero only in the boundary incident to its dimension (i.e. $\psi_x$ is only non-zero in the $x$-direction boundaries). This simplifies the equations in most of the boundary region: the full equation needs to be computed only for boundary corners.

The full set of equations with boundary conditions then becomes:
$$
\begin{align}
\frac{\partial^2 p}{\partial t^2} &= c^2 \nabla^2 p + s &&,~ \text{in}~ \overline{\Omega} &&\times [0,T], \\
\frac{\partial^2 p}{\partial t^2} &= c^2 \left( \nabla^2 p + \left( \frac{\partial \psi_i}{\partial i} \right) + \xi_i \right)&&,~ \text{in}~ \tilde{\partial}_i \Omega &&\times [0,T], \\
p &= 0 &&,~ \text{in}~ \partial \Omega &&\times [0,T]
\end{align}
$$

## Numerical methods and implementation

We used the most simple finite-difference scheme: second order explicit FD schemes for both space and time. We stagger the $\psi_i$ field in between grid nodal points in $\tilde{\partial}_i \Omega$ and place $p, c, \xi_i$ fields in nodal points ($\xi_i$ only present in $\tilde{\partial}_i \Omega$).

We split a time step computations into two substeps:
- first we update the $\psi_i$ fields using $(1)$,
- then we update the $\xi_i$ fields using $(2)$ at the same time we use $(3)$ and $(4)$ to update pressure values.

This separation is crucial to ensure that the $\psi_i$ fields are computed before updating pressure with $(4)$. Syncronization is used to make sure this happens. Note that, since every $\psi_i$ field in each dimension $i$ is independent of each other and only uses pressure derivatives in the direction of indicidence of $\tilde{\partial_i} \Omega$. This means that we dont have any inter-depencencies between different dimensions, so we can compute all $\psi_i$ fields in parallel.

## Results

We have run several simulations in 1D, 2D and 3D to ensure qualitative analysis of CPML boundary conditions in variuos setups. In this section, we briefly explain the setups used.

### Setups

For the source terms, we always use a Ricker wavelet source (second drivative of a Gaussian) to simulate an "impulse" source. We used sources at $8$ Hz and we scale them in amplitude for the 2D and 3D setups (this is done because we need a "stronger" source in 2D and 3D because the energy is spread over an increasingly bigger volume).

We also scale the model size in order to always have at least 10 points per wavelength to deal with numerical dipersion. For simplicity, we adpted a constant grid step size in all dimensions for all setups. This does not mean that the grid point number is the same everytime, it just means that the grid cells are square and always have the same size in all dimensions (although our code in general and can be used for any grid step size). This is the same for time step size, we have fixed one that works and fulfill the CFL stability criterion (but one can choose it arbitrarely).

We have mainly three types of setups:
- constant velocity models with a single source located in the center and CPML layers in all boundary regions;
- gradient velocity (directly proportional to depth) with multiple sources located at the top of the model and **NO** CPML layers on the top boundary (to simulate free top boundary conditions).
- complex velocity model 

### 1D CPML

### 2D CPML

### 3D CPML

## Performance evaluation

Talk about the setup for benchmarks (Piz Daint, my laptop etc...). Mention peak performances and other relevant hardware/software information.

### Performance of 2D and 3D kernels

Show performance plot.

### Weak scaling of 2D and 3D multi-xPU kernels

Show weak scaling plot.

## Conclusions

Talk about what has been achieved and what has been planned for the future.

## Reference

Add references (CPML, complex model, etc...)
