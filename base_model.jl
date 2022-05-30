using Oceananigans
using Oceananigans.Models.ShallowWaterModels: VectorInvariantFormulation
using Oceananigans.Advection: VelocityStencil, VorticityStencil


## Define 2D grid

Lx, Ly, Lz = 2π, 20, 1
Nx, Ny = 128, 128

grid = RectilinearGrid(size = (Nx, Ny), x = (0, Lx), y = (-Ly/2, Ly/2), topology = (Periodic, Bounded, Flat))

## Forcing functions for the SWMHD model

using Oceananigans.Operators: ℑxᶜᵃᵃ, ∂xᶠᶜᶜ, ℑyᵃᶜᵃ, ∂yᶜᶠᶜ

# Computes ∂x(A)/h
function x_inner_func(i, j, k, grid, clock, fields)
    return ℑxᶜᵃᵃ(i, j, k, grid, ∂xᶠᶜᶜ, fields.A)/fields.h[i, j, k]
end

# Computes ∂y(A)/h
function y_inner_func(i, j, k, grid, clock, fields)
    return ℑyᵃᶜᵃ(i, j, k, grid, ∂yᶜᶠᶜ, fields.A)/fields.h[i, j, k]
end

# Computes the Jacobian of A and ∂x(A)/h, the x-component of the Jacobian
function jacobian_x(i, j, k, grid, clock, fields)
    return ℑxᶜᵃᵃ(i, j, k, grid, ∂xᶠᶜᶜ, fields.A) * ℑyᵃᶜᵃ(i, j, k, grid, ∂yᶜᶠᶜ, x_inner_func(i, j, k, grid, clock, fields)) 
    - ℑyᵃᶜᵃ(i, j, k, grid, ∂yᶜᶠᶜ, fields.A) * ℑxᶜᵃᵃ(i, j, k, grid, ∂xᶠᶜᶜ, x_inner_func(i, j, k, grid, clock, fields))
end

# Computes the Jacobian of A and ∂y(A)/h, the y-component of the Jacobian
function jacobian_y(i, j, k, grid, clock, fields)
    return ℑxᶜᵃᵃ(i, j, k, grid, ∂xᶠᶜᶜ, fields.A) * ℑyᵃᶜᵃ(i, j, k, grid, ∂yᶜᶠᶜ, y_inner_func(i, j, k, grid, clock, fields)) 
    - ℑyᵃᶜᵃ(i, j, k, grid, ∂yᶜᶠᶜ, fields.A) * ℑxᶜᵃᵃ(i, j, k, grid, ∂xᶠᶜᶜ, y_inner_func(i, j, k, grid, clock, fields))
end

# Computes the u-component forcing term; note that jacobian_y() is used because 
# the cross product of -ẑ and ŷ is x̂
function u_forcing_func(i, j, k, grid, clock, fields)
    return (1/fields.h[i, j, k])*(jacobian_y(i, j, k, grid, clock, fields))
end

# Computes the v-component forcing term; note that jacobian_x() is used because
# the cross product of -ẑ and x̂ is -ŷ
function v_forcing_func(i, j, k, grid, clock, fields)
    return (-1/fields.h[i, j, k])*(jacobian_x(i, j, k, grid, clock, fields))
end

u_forcing = Forcing(u_forcing_func, discrete_form = true)

v_forcing = Forcing(v_forcing_func, discrete_form = true)

## Model parameters (variable)

const U = 1.0         # Maximum jet velocity

f = 1000         # Coriolis parameter
g = 10^-3         # Gravitational acceleration
Δη = f * U / g  # Maximum free-surface deformation as dictated by geostrophy

## Construction of SWMHD model

model = ShallowWaterModel(
                          timestepper = :RungeKutta3,
                          momentum_advection = WENO5(vector_invariant = VelocityStencil()),
                          grid = grid,
                          gravitational_acceleration = g,
                          coriolis = FPlane(f=f),
                          tracers = (:A),
                          forcing = (u = u_forcing, v = v_forcing),
                          formulation = VectorInvariantFormulation())

## Setup background state and perturbation

## Setup initial conditins

## Run simulation

# Determine Δt through analysis of time scale needed
# simulation = Simulation(model, Δt = 1e-2, stop_time = 250)

# run!(simulation)

## Visualize results

