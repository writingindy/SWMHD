using Oceananigans
using Oceananigans.Models.ShallowWaterModels: VectorInvariantFormulation
using Oceananigans.Advection: VelocityStencil, VorticityStencil
using Oceananigans.AbstractOperations: UnaryOperation
using Oceananigans.Operators
using CairoMakie, Statistics, JLD2, Printf, NCDatasets

include("sw_mhd_jacobian_functions.jl")

const Lx = 10
const Ly = 10
const Nx = 512
const Ny = 512
grid = RectilinearGrid(GPU(); size = (Nx, Ny), 
                           x = (-Lx/2, Lx/2), y = (-Ly/2, Ly/2),
                   topology = (Periodic, Periodic, Flat))

filename = "jacobian_two_Gaussians_high_B"
energies_filename = "jacobian_two_Gaussians_high_B_energies"

model = ShallowWaterModel(grid = grid,
                          timestepper = :RungeKutta3,
                          momentum_advection = WENO5(vector_invariant = VelocityStencil()),
                          mass_advection = WENO5(),
                          tracer_advection = WENO5(),
                          gravitational_acceleration = 9.81,
                          coriolis = FPlane(f=1),
                          tracers = (:A),
                          forcing = (u = Forcing(lorentz_force_func_x, discrete_form = true), 
                                     v = Forcing(lorentz_force_func_y, discrete_form = true)),
                          formulation = VectorInvariantFormulation()
                          )

@info "Model instantiated!"

Aᵢ(x, y, z) = 0.5*exp(-((x - 0.5)^2 + y^2)) - 0.5*exp(-((x + 0.5)^2 + y^2))
hᵢ(x, y, z) = 1
set!(model, h = hᵢ, A = Aᵢ)
simulation = Simulation(model, Δt = 0.001, stop_time = 80)

@info "Simulation instantiated!"


u, v, h = model.solution
A = model.tracers.A
s = sqrt(u^2 + v^2)
compute!(s)


B_x = Field(-∂y(A)/h)
B_y =  Field(∂x(A)/h)
compute!(B_x)
compute!(B_y)

kinetic_energy_func   = Integral((1/2)*(h)*(u^2 + v^2))
potential_energy_func = Integral((1/2)*model.gravitational_acceleration*(h - hᵢ)^2)

simulation.output_writers[:fields] = JLD2OutputWriter(model, (; h, A = model.tracers.A, s, B_x, B_y),
                                                      schedule = TimeInterval(0.1),
                                                      filename = filename * ".jld2",
                                                      overwrite_existing = true)

simulation.output_writers[:energies] = JLD2OutputWriter(model, (; kinetic_energy_func, potential_energy_func),
                                                      schedule = TimeInterval(0.1),
                                                      filename = energies_filename * ".jld2",
                                                      overwrite_existing = true)



@info "Running with Δt = $(prettytime(simulation.Δt))"
sim_start_time = time_ns()*1e-9
run!(simulation)
sim_end_time = time_ns()*1e-9
sim_time = prettytime(sim_end_time - sim_start_time)
@info "Simulation took $(sim_time) to finish running."

