using Oceananigans
using Oceananigans.Models.ShallowWaterModels: ConservativeFormulation
using Oceananigans.Advection
using Oceananigans.Operators
using Oceananigans.Grids: AbstractGrid, topology
using CairoMakie, Statistics, JLD2, Printf, NCDatasets, Polynomials


const Lx = 2π
const Ly = 20
const Lz = 10
const Nx = 128
const Ny = 128

grid = RectilinearGrid(; size = (Nx, Ny), 
                           x = (0, Lx), y = (-Ly/2, Ly/2),
                   topology = (Periodic, Bounded, Flat))

filename = "divergence_bickley_jet"
energies_filename = "divergence_bickley_jet_energies"
output_prefix = "divergence_bickley_jet"

gravitational_acceleration = 1
coriolis = FPlane(f=1)

model = ShallowWaterModel(;grid, 
                          timestepper = :RungeKutta3,
                          momentum_advection = WENO5(),
                          gravitational_acceleration = 1,
                          coriolis = FPlane(f=1),
                          formulation = ConservativeFormulation())


U = 1 # Maximum jet velocity
f = coriolis.f
g = gravitational_acceleration
Δη = f * U / g  # Maximum free-surface deformation as dictated by geostrophy

h̄(x, y, z) = Lz - Δη * tanh(y)
ū(x, y, z) = U * sech(y)^2

ω̄(x, y, z) = 2 * U * sech(y)^2 * tanh(y)

small_amplitude = 1e-4

 uⁱ(x, y, z) = ū(x, y, z) + small_amplitude * exp(-y^2) * randn()
uhⁱ(x, y, z) = uⁱ(x, y, z) * h̄(x, y, z)

ū̄h(x, y, z) = ū(x, y, z) * h̄(x, y, z)

set!(model, uh = ū̄h, h = h̄)

uh, vh, h = model.solution

set!(model, uh = uhⁱ)

kinetic_energy_func   = Integral((1/2)*(1/h)*(uh^2 + vh^2))
potential_energy_func = Integral((1/2)*model.gravitational_acceleration*(h)^2)

simulation = Simulation(model, Δt = 1e-2, stop_time = 150)

simulation.output_writers[:energies] = JLD2OutputWriter(model, (; kinetic_energy_func, potential_energy_func),
                                                      schedule = TimeInterval(0.1),
                                                      filename = energies_filename * ".jld2",
                                                      overwrite_existing = true)



run!(simulation)   

@info "Making a plot of the various energies of the system..."

KE_timeseries = FieldTimeSeries(energies_filename * ".jld2", "kinetic_energy_func")
PE_timeseries = FieldTimeSeries(energies_filename * ".jld2", "potential_energy_func")

t = KE_timeseries.times

total_energy = zeros(length(t))

kinetic_energy = interior(KE_timeseries)[1, 1, 1, :]
potential_energy = interior(PE_timeseries)[1, 1, 1, :]

for i in 1:length(t)
    total_energy[i] = kinetic_energy[i] + potential_energy[i]
end

initial_total_energy = first(total_energy)
deviation_total_energy = (abs.(total_energy .- initial_total_energy) ./ initial_total_energy) .* 100


n = Observable(1)

f = Figure()

Axis(f[1, 1], title = "kinetic energy")
lines!(t, kinetic_energy; linewidth = 4, color = "red")

Axis(f[2, 1], title = "potential energy")
lines!(t, potential_energy; linewidth = 4, color = "green")

Axis(f[2, 2], title = "relative energy error (%)")
lines!(t, deviation_total_energy; linewidth = 4, color = "black")

Label(f[0, :], "128x128 Bickley Jet: Energy Plots", textsize = 20)

save("divergence_bickley_jet_energy_plot.png", f)