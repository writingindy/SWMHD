using Oceananigans
using Oceananigans.Models.ShallowWaterModels: VectorInvariantFormulation
using Oceananigans.Advection: VelocityStencil, VorticityStencil
using Oceananigans.Operators
using CairoMakie, Statistics, JLD2, Printf

include("sw_mhd_jacobian_functions.jl")

Lx, Ly = 10, 10
Nx, Ny = 128, 128


grid = RectilinearGrid(size = (Nx, Ny), 
                          x = (-Lx/2, Lx/2), y = (-Ly/2, Ly/2), 
                   topology = (Periodic, Periodic, Flat))

using Oceananigans.TurbulenceClosures
#horizontal_diffusivity = HorizontalScalarDiffusivity(ν=νh)
biharmonic_viscosity   = HorizontalScalarBiharmonicDiffusivity(ν=1e-9, κ=1e-9)

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
                          #closure = biharmonic_viscosity,
                          formulation = VectorInvariantFormulation()
                          )



Aᵢ(x, y, z) = 0.5*exp(-((x - 0.5)^2 + y^2)) - 0.5*exp(-((x + 0.5)^2 + y^2))
#uᵢ(x, y, z) = y*exp(-(x^2 + y^2))
#vᵢ(x, y, z) = -x*exp(-(x^2 + y^2))
set!(model,#= u = uᵢ, v = vᵢ,=# h = 1, A = Aᵢ)
simulation = Simulation(model, Δt = 0.01, stop_time = 25.0)

u, v, h = simulation.model.solution
A = simulation.model.tracers.A
B_x = -∂y(A) / h
B_y = ∂x(A) / h
kinetic_energy = h*(u^2 + v^2)
magnetic_energy = h*(B_x^2 + B_y^2)
potential_energy = model.gravitational_acceleration*h^2
total_energy = compute!(kinetic_energy + magnetic_energy + potential_energy)
initial_energy = (1/2)*sum(total_energy[1:Nx, 1:Ny])



start_time = [time_ns()]

function progress(sim)
    wall_time = (time_ns() - start_time[1]) * 1e-9

    u = sim.model.solution.u
    h = sim.model.solution.h
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

u, v, h = model.solution
s = sqrt(u^2 + v^2)
compute!(s)

filename = "SW_MHD_adjustment"

simulation.output_writers[:fields] = JLD2OutputWriter(model, (; u, v, A = model.tracers.A, s),
                                                      schedule = TimeInterval(0.1),
                                                      filename = filename * ".jld2",
                                                      overwrite_existing = true)

@info "Running with Δt = $(prettytime(simulation.Δt))"
sim_start_time = time_ns()*1e-9
run!(simulation)
sim_end_time = time_ns()*1e-9
sim_time = prettytime(sim_end_time - sim_start_time)
@info "Simulation took $(sim_time) to finish running."

u, v, h = simulation.model.solution
A = simulation.model.tracers.A
B_x = -∂y(A) / h
B_y = ∂x(A) / h
kinetic_energy = h*(u^2 + v^2)
magnetic_energy = h*(B_x^2 + B_y^2)
potential_energy = model.gravitational_acceleration*h^2
total_energy = compute!(kinetic_energy + magnetic_energy + potential_energy)
final_energy = (1/2)*sum(total_energy[1:Nx, 1:Ny])

relative_energy_error = abs(final_energy - initial_energy)/initial_energy

@info "Percentage difference in total energy is $(round(relative_energy_error * 100, digits = 3))%"


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
