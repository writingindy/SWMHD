using Oceananigans
using Oceananigans.Models.ShallowWaterModels: ConservativeFormulation
using Oceananigans.Advection
using Oceananigans.Operators
using Oceananigans.Grids: AbstractGrid, topology
using CairoMakie, Statistics, JLD2, Printf, NCDatasets

include("sw_mhd_divergence_functions.jl")

const Lx = 10
const Ly = 10
const Nx = 512
const Ny = 512
grid = RectilinearGrid(GPU(); size = (Nx, Ny), 
                           x = (-Lx/2, Lx/2), y = (-Ly/2, Ly/2),
                   topology = (Periodic, Bounded, Flat))

filename = "divergence_low_B_low_U"
energies_filename = "divergence_low_B_low_U_energies"

A_bcs = FieldBoundaryConditions(north = GradientBoundaryCondition(-0.05), south = GradientBoundaryCondition(-0.05))

model = ShallowWaterModel(grid = grid,
                          timestepper = :RungeKutta3,
                          boundary_conditions = (A = A_bcs, ),
                          momentum_advection = WENO5(),
                          mass_advection = WENO5(),
                          tracer_advection = WENO5(),
                          gravitational_acceleration = 9.81,
                          coriolis = FPlane(f=1),
                          tracers = (:A),
                          forcing = (uh = Forcing(div_lorentz_x, discrete_form = true), 
                                     vh = Forcing(div_lorentz_y, discrete_form = true)),
                          formulation = ConservativeFormulation()
                          )

@info "Model instantiated!"

#Aᵢ(x, y, z) = 0.5*exp(-((x - 0.5)^2 + y^2)) - 0.5*exp(-((x + 0.5)^2 + y^2))
Aᵢ(x, y, z) = -0.05*y
hᵢ(x, y, z) = 1
uhᵢ(x, y, z) = y*exp(-(x^2 + y^2))
vhᵢ(x, y, z) = -x*exp(-(x^2 + y^2))
set!(model, uh = uhᵢ, vh = vhᵢ, h = hᵢ, A = Aᵢ)
simulation = Simulation(model, Δt = 0.001, stop_time = 30)

@info "Simulation instantiated!"


uh, vh, h = model.solution
A = model.tracers.A
u = uh / h
v = vh / h
s = sqrt(u^2 + v^2)
compute!(s)


B_x = Field(-∂y(A)/h)
B_y =  Field(∂x(A)/h)
compute!(B_x)
compute!(B_y)

kinetic_energy_func   = Integral((1/2)*(1/h)*(uh^2 + vh^2))
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

