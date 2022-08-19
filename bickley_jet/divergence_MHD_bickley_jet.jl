using Oceananigans
using Oceananigans.Models.ShallowWaterModels: ConservativeFormulation
using Oceananigans.Advection
using Oceananigans.Operators
using Oceananigans.Grids: AbstractGrid, topology
using CairoMakie, Statistics, JLD2, Printf, NCDatasets, Polynomials


const Lx = 8π
const Ly = 8π
const Lz = 10
const Nx = 256
const Ny = 256

include("sw_mhd_divergence_functions.jl")

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
                          tracers = (:A),
                          forcing = (uh = Forcing(div_lorentz_x, discrete_form = true), 
                                     vh = Forcing(div_lorentz_y, discrete_form = true)),
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

# Build velocities
u = uh / h
v = vh / h

Aᵢ(x, y, z) = -0.05*y

# Build and compute mean vorticity discretely
ω = Field(∂x(v) - ∂y(u))
compute!(ω)

# Copy mean vorticity to a new field
ωⁱ = Field{Face, Face, Nothing}(model.grid)
ωⁱ .= ω

# Use this new field to compute the perturbation vorticity
ω′ = Field(ω - ωⁱ)

set!(model, uh = uhⁱ, A = Aᵢ)


A = model.tracers.A

hB_x = -∂y(A)
hB_y =  ∂x(A)

kinetic_energy_func   = Integral((1/2)*(1/h)*(uh^2 + vh^2))
magnetic_energy_func  = Integral((1/2)*(1/h)*(hB_x^2 + hB_y^2))
potential_energy_func = Integral((1/2)*model.gravitational_acceleration*(h)^2)

simulation = Simulation(model, Δt = 5e-3, stop_time = 150)

fields_filename = joinpath(@__DIR__, "divergence_bickley_jet_fields.nc")
simulation.output_writers[:fields] = NetCDFOutputWriter(model, (; ω, ω′),
                                                        filename = fields_filename,
                                                        schedule = TimeInterval(1),
                                                        overwrite_existing = true)

simulation.output_writers[:energies] = JLD2OutputWriter(model, (; kinetic_energy_func, magnetic_energy_func, potential_energy_func),
                                                      schedule = TimeInterval(0.1),
                                                      filename = energies_filename * ".jld2",
                                                      overwrite_existing = true)



run!(simulation)   

@info "Making animation of total and perturbation vorticity"

x, y = xnodes(ω), ynodes(ω)

fig = Figure(resolution = (1200, 660))

axis_kwargs = (xlabel = "x",
               ylabel = "y",
               aspect = AxisAspect(1),
               limits = ((0, Lx), (-Ly/2, Ly/2)))

ax_ω  = Axis(fig[2, 1]; title = "Total vorticity, ω", axis_kwargs...)
ax_ω′ = Axis(fig[2, 3]; title = "Perturbation vorticity, ω - ω̄", axis_kwargs...)

n = Observable(1)

ds = NCDataset(simulation.output_writers[:fields].filepath, "r")

times = ds["time"][:]

ω = @lift ds["ω"][:, :, 1, $n]
hm_ω = heatmap!(ax_ω, x, y, ω, colorrange = (-1, 1), colormap = :balance)
Colorbar(fig[2, 2], hm_ω)

ω′ = @lift ds["ω′"][:, :, 1, $n]
hm_ω′ = heatmap!(ax_ω′, x, y, ω′, colormap = :balance)
Colorbar(fig[2, 4], hm_ω′)

title = @lift @sprintf("t = %.1f", times[$n])
fig[1, 1:4] = Label(fig, title, textsize=24, tellwidth=false)

frames = 1:length(times)

record(fig, "divergence_bickley_jet.mp4", frames, framerate=12) do i
    msg = string("Plotting frame ", i, " of ", frames[end])
    print(msg * " \r")
    n[] = i
end

close(ds)

@info "Making a plot of the various energies of the system..."

KE_timeseries = FieldTimeSeries(energies_filename * ".jld2", "kinetic_energy_func")
ME_timeseries = FieldTimeSeries(energies_filename * ".jld2", "magnetic_energy_func")
PE_timeseries = FieldTimeSeries(energies_filename * ".jld2", "potential_energy_func")

t = KE_timeseries.times

total_energy = zeros(length(t))

kinetic_energy = interior(KE_timeseries)[1, 1, 1, :]
magnetic_energy = interior(ME_timeseries)[1, 1, 1, :]
potential_energy = interior(PE_timeseries)[1, 1, 1, :]

for i in 1:length(t)
    total_energy[i] = kinetic_energy[i] + potential_energy[i] + magnetic_energy[i]
end

initial_total_energy = first(total_energy)
deviation_total_energy = (abs.(total_energy .- initial_total_energy) ./ initial_total_energy) .* 100


n = Observable(1)

f = Figure()

Axis(f[1, 1], title = "kinetic energy")
lines!(t, kinetic_energy; linewidth = 4, color = "red")

Axis(f[1, 2], title = "magnetic energy")
lines!(t, magnetic_energy; linewidth = 4, color = "blue")

Axis(f[2, 1], title = "potential energy")
lines!(t, potential_energy; linewidth = 4, color = "green")

Axis(f[2, 2], title = "relative energy error (%)")
lines!(t, deviation_total_energy; linewidth = 4, color = "black")

Label(f[0, :], "256x256 Bickley Jet: Energy Plots", textsize = 20)

save("divergence_bickley_jet_energy_plot.png", f)