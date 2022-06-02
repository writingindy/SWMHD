using Oceananigans
using Oceananigans.Models.ShallowWaterModels: VectorInvariantFormulation
using Oceananigans.Advection: VelocityStencil, VorticityStencil
using Oceananigans.Operators: ℑxᶜᵃᵃ, ∂xᶠᶜᶜ, ℑyᵃᶜᵃ, ∂yᶜᶠᶜ, ℑxyᶠᶜᵃ, ℑxyᶜᶠᵃ, ℑxᶠᵃᵃ, ℑyᵃᶠᵃ
using NCDatasets, Plots, Printf

# Computes ∂x(A)/h at ccc
# A and h are at centers, ccc
# ∂x(A) is at fcc, and we interpolate it to ccc
function ∂xA_over_h(i, j, k, grid, clock, fields)
    return ℑxᶜᵃᵃ(i, j, k, grid, ∂xᶠᶜᶜ, fields.A)/fields.h[i, j, k]
end

# Computes ∂y(A)/h at ccc
# A and h are at centers, ccc
# ∂y(A) is at cfc, and we interpolate it to ccc
function ∂yA_over_h(i, j, k, grid, clock, fields)
    return ℑyᵃᶜᵃ(i, j, k, grid, ∂yᶜᶠᶜ, fields.A)/fields.h[i, j, k]
end

# Computes the Jacobian at fcc for the u-component forcing term
function jacobian_x(i, j, k, grid, clock, fields)
    return ∂xᶠᶜᶜ(i, j, k, grid, fields.A) * ℑxyᶠᶜᵃ(i, j, k, grid, ∂yᶜᶠᶜ, ∂yA_over_h(i, j, k, grid, clock, fields))
           - ℑxyᶠᶜᵃ(i, j, k, grid, ∂yᶜᶠᶜ, fields.A) * ∂xᶠᶜᶜ(i, j, k, grid, ∂yA_over_h(i, j, k, grid, clock, fields))
end


# Computes the Jacobian at cfc for the v-component forcing term
function jacobian_y(i, j, k, grid, clock, fields)
    return ℑxyᶜᶠᵃ(i, j, k, grid, ∂xᶠᶜᶜ, fields.A) * ∂yᶜᶠᶜ(i, j, k, grid, ∂xA_over_h(i, j, k, grid, clock, fields))
           - ∂yᶜᶠᶜ(i, j, k, grid, fields.A) * ℑxyᶜᶠᵃ(i, j, k, grid, ∂xᶠᶜᶜ, ∂xA_over_h(i, j, k, grid, clock, fields))
end

# Computes the u-component forcing term at fcc 
# Note that jacobian_y() is used because -ẑ × ŷ = x̂
function Lorentz_forcing_term_x(i, j, k, grid, clock, fields)
    return (1/ℑxᶠᵃᵃ(i, j, k, grid, fields.h))*(jacobian_x(i, j, k, grid, clock, fields))
end

# Computes the v-component forcing term at cfc; 
# Note that jacobian_x() is used because -ẑ × x̂ = -ŷ
function Lorentz_forcing_term_y(i, j, k, grid, clock, fields)
    return (-1/ℑyᵃᶠᵃ(i, j, k, grid, fields.h))*(jacobian_y(i, j, k, grid, clock, fields))
end


# Model parameters

Lx, Ly, Lz = 2π, 20, 1
Nx, Ny = 128, 128

const U = 1.0         # Maximum jet velocity

f = 1         # Coriolis parameter
g = 9.81         # Gravitational acceleration
Δη = f * U / g  # Maximum free-surface deformation as dictated by geostrophy

grid = RectilinearGrid(size = (Nx, Ny), x = (0, Lx), y = (-Ly/2, Ly/2), topology = (Periodic, Bounded, Flat))

## Forcing functions for the SWMHD model

Lorentz_force_x = Forcing(Lorentz_forcing_term_x, discrete_form = true)
Lorentz_force_y = Forcing(Lorentz_forcing_term_y, discrete_form = true)

## Construction of SWMHD model

model = ShallowWaterModel(
                          timestepper = :RungeKutta3,
                          momentum_advection = WENO5(vector_invariant = VelocityStencil()),
                          grid = grid,
                          gravitational_acceleration = g,
                          coriolis = FPlane(f=f),
                          tracers = (:A),
                          forcing = (u = Lorentz_force_x, v = Lorentz_force_y),
                          formulation = VectorInvariantFormulation())

## Background state and perturbation

h̄(x, y, z) = Lz - Δη * tanh(y)
ū(x, y, z) = U * sech(y)^2
ω̄(x, y, z) = 2 * U * sech(y)^2 * tanh(y)

small_amplitude = 1e-4

uⁱ(x, y, z) = ū(x, y, z) + small_amplitude * exp(-y^2) * randn()

Aᵢ(x, y, z) = -y + randn()

set!(model, u = ū, h = h̄, A = Aᵢ)

u, v, h = model.solution

ω = Field(∂x(v) - ∂y(u))
compute!(ω)

ωⁱ = Field{Face, Face, Nothing}(model.grid)
ωⁱ .= ω

ω′ = Field(ω - ωⁱ)

set!(model, u = uⁱ)

## Running the simulation

simulation = Simulation(model, Δt = 1e-2, stop_time = 250)

using LinearAlgebra: norm

perturbation_norm(args...) = norm(v)

simulation.output_writers[:fields] = NetCDFOutputWriter(model, (; ω, ω′),
                                                        filename = joinpath(@__DIR__, "shallow_water_Bickley_jet_fields.nc"),
                                                        schedule = TimeInterval(1),
                                                        overwrite_existing = true)

simulation.output_writers[:growth] = NetCDFOutputWriter(model, (; perturbation_norm),
                                                        filename = joinpath(@__DIR__, "shallow_water_Bickley_jet_perturbation_norm.nc"),
                                                        schedule = IterationInterval(1),
                                                        dimensions = (; perturbation_norm = ()),
                                                        overwrite_existing = true)

run!(simulation)

## Visualizing the results

x, y = xnodes(ω), ynodes(ω)

kwargs = (
         xlabel = "x",
         ylabel = "y",
         aspect = 1,
           fill = true,
         levels = 20,
      linewidth = 0,
          color = :balance,
       colorbar = true,
           ylim = (-Ly/2, Ly/2),
           xlim = (0, Lx)
)

ds = NCDataset(simulation.output_writers[:fields].filepath, "r")

anim = @animate for (iter, t) in enumerate(ds["time"])
    ω = ds["ω"][:, :, 1, iter]
    ω′ = ds["ω′"][:, :, 1, iter]

    ω′_max = maximum(abs, ω′)

    plot_ω = contour(x, y, ω',
                     clim = (-1, 1),
                     title = @sprintf("Total vorticity, ω, at t = %.1f", t); kwargs...)

    plot_ω′ = contour(x, y, ω′',
                      clim = (-ω′_max, ω′_max),
                      title = @sprintf("Perturbation vorticity, ω - ω̄, at t = %.1f", t); kwargs...)

    plot(plot_ω, plot_ω′, layout = (1, 2), size = (800, 440))
end

close(ds)

mp4(anim, "shallow_water_Bickley_jet.mp4", fps=15)
