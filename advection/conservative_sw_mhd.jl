using Oceananigans
using Oceananigans.Models.ShallowWaterModels: ConservativeFormulation
using Oceananigans.Advection
using Oceananigans.Operators
using Oceananigans.Grids: AbstractGrid
using CairoMakie, Statistics, JLD2, Printf, NCDatasets

include("conservative_advection_functions.jl")

Lx, Ly = 10, 10
Nx, Ny = 64, 64

grid = RectilinearGrid(size = (Nx, Ny), 
                          x = (-Lx/2, Lx/2), y = (-Ly/2, Ly/2), 
                   topology = (Periodic, Periodic, Flat))

model = ShallowWaterModel(grid = grid,
                          timestepper = :RungeKutta3,
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

#Aᵢ(x, y, z) = 0.1*exp(-((x - 0.5)^2 + y^2)) - 0.1*exp(-((x + 0.5)^2 + y^2))
Aᵢ(x, y, z) = -0.05*y
hᵢ(x, y, z) = 1
uhᵢ(x, y, z) = y*exp(-(x^2 + y^2))
vhᵢ(x, y, z) = -x*exp(-(x^2 + y^2))
set!(model, uh = uhᵢ, vh = vhᵢ, h = hᵢ, A = Aᵢ)
simulation = Simulation(model, Δt = 0.01, stop_time = 25)

start_time = [time_ns()]
function progress(sim)
    wall_time = (time_ns() - start_time[1]) * 1e-9

    uh = sim.model.solution.uh
    h = sim.model.solution.h
    u = uh / h
    A = sim.model.tracers.A 

    @info @sprintf("Time: % 12s, iteration: %d, max(|u|): %.2e ms⁻¹, max(A): %.2e ms⁻¹, min(h): %.2e ms⁻¹, wall time: %s",
                    prettytime(sim.model.clock.time),
                    sim.model.clock.iteration, 
                    maximum(abs, u), maximum(abs, A), minimum(h),
                    prettytime(wall_time))

    start_time[1] = time_ns()

    return nothing
end

simulation.callbacks[:progress] = Callback(progress, IterationInterval(1))

uh, vh, h = model.solution
u = uh / h
v = vh / h
s = sqrt(u^2 + v^2)
A = model.tracers.A
B_x = -∂y(A) / h
B_y = ∂x(A) / h
kinetic_energy_func(args...) = (1/2)*sum(h*(u^2 + v^2))*grid.Δxᶜᵃᵃ*grid.Δyᵃᶜᵃ
magnetic_energy_func(args...) = (1/2)*sum(h*(B_x^2 + B_y^2))*grid.Δxᶜᵃᵃ*grid.Δyᵃᶜᵃ
potential_energy_func(args...) = (1/2)*sum(model.gravitational_acceleration*h^2)*grid.Δxᶜᵃᵃ*grid.Δyᵃᶜᵃ
total_energy_func(args...) = (1/2)*sum(h*(u^2 + v^2))*grid.Δxᶜᵃᵃ*grid.Δyᵃᶜᵃ + (1/2)*sum(h*(B_x^2 + B_y^2))*grid.Δxᶜᵃᵃ*grid.Δyᵃᶜᵃ + (1/2)*sum(model.gravitational_acceleration*h^2)*grid.Δxᶜᵃᵃ*grid.Δyᵃᶜᵃ
compute!(s)

filename = "SW_MHD_adjustment"
simulation.output_writers[:fields] = JLD2OutputWriter(model, (; u, v, A = model.tracers.A, s),
                                                      schedule = TimeInterval(0.1),
                                                      filename = filename * ".jld2",
                                                      overwrite_existing = true)

energies_filename = joinpath(@__DIR__, "energies.nc")
simulation.output_writers[:energies] = NetCDFOutputWriter(model, (; kinetic_energy_func, magnetic_energy_func, potential_energy_func, total_energy_func),
                                                        filename = energies_filename,
                                                        schedule = IterationInterval(1),
                                                        dimensions = (; kinetic_energy_func = (), magnetic_energy_func = (), potential_energy_func = (), total_energy_func = ()),
                                                        overwrite_existing = true)


@info "Running with Δt = $(prettytime(simulation.Δt))"
sim_start_time = time_ns()*1e-9
run!(simulation)
sim_end_time = time_ns()*1e-9
sim_time = prettytime(sim_end_time - sim_start_time)
@info "Simulation took $(sim_time) to finish running."

output_prefix = "SW_MHD_adjustment"

x, y, z = nodes((Center, Center, Center), grid)
s_timeseries = FieldTimeSeries(filename * ".jld2", "s")
A_timeseries = FieldTimeSeries(filename * ".jld2", "A")
times = s_timeseries.times

@info "Making a movie of the magnetic potential function A..."

iter = Observable(2)
A = @lift interior(A_timeseries[$iter], :, :, 1)
s = @lift interior(s_timeseries[$iter], :, :, 1)
title_A = @lift(@sprintf("Magnetic potential at time = %s", string(round(times[$iter], digits = 2))))
title_s = @lift(@sprintf("Speed at time = %s", string(round(times[$iter], digits = 2))))
fig = Figure(resolution = (800, 400))
ax_A = Axis(fig[1,1], xlabel = "x", ylabel = "y", title=title_A)
ax_s = Axis(fig[1,2], xlabel = "x", ylabel = "y", title=title_s)
heatmap!(ax_A, x, y, A, colormap=:deep)
heatmap!(ax_s, x, y, s, colormap=:deep)

frames = 2:length(times)

record(fig, output_prefix * ".mp4", frames, framerate=96) do i
    @info "Plotting iteration $i of $(frames[end])..."
    iter[] = i
end

@info "Making a plot of the various energies of the system..."

ds2 = NCDataset(simulation.output_writers[:energies].filepath, "r")

                t = ds2["time"][:]
   kinetic_energy = ds2["kinetic_energy_func"][:]
  magnetic_energy = ds2["magnetic_energy_func"][:]
 potential_energy = ds2["potential_energy_func"][:]
     total_energy = ds2["total_energy_func"][:]

close(ds2)

f = Figure()
ax = Axis(f[1, 1], xlabel = "time", ylabel = "energy", title = "Plot of different energies")

lines!(t, kinetic_energy; linewidth = 4, label = "kinetic energy",)
lines!(t, magnetic_energy; linewidth = 4, label = "magnetic energy")
lines!(t, potential_energy; linewidth = 4, label = "potential energy")
lines!(t, total_energy; linewidth = 4, label = "total energy")
axislegend()

save("energy_plot.png", f)

final_total_energy = last(total_energy)
initial_total_energy = first(total_energy)
relative_energy_error = abs(final_total_energy - initial_total_energy)/initial_total_energy

@info "Percentage difference in total energy is $(relative_energy_error * 100)%"