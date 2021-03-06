using Oceananigans
using Oceananigans.Models.ShallowWaterModels: VectorInvariantFormulation
using Oceananigans.Advection: VelocityStencil, VorticityStencil
using Oceananigans.AbstractOperations: UnaryOperation
using Oceananigans.Operators
using CairoMakie, Statistics, JLD2, Printf, NCDatasets

include("sw_mhd_jacobian_functions.jl")

Lx, Ly = 10, 10
Nx, Ny = 64, 64


grid = RectilinearGrid(size = (Nx, Ny), 
                          x = (-Lx/2, Lx/2), y = (-Ly/2, Ly/2), 
                   topology = (Periodic, Periodic, Flat))

#north_bcs = 
#A_bcs = FieldBoundaryConditions(north = GradientBoundaryCondition(-0.05), south = GradientBoundaryCondition(-0.05))

model = ShallowWaterModel(grid = grid,
                          #boundary_conditions = (A = A_bcs, ),
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


Aᵢ(x, y, z) = 0.5*abs(y)
#Aᵢ(x, y, z) = 0.1*exp(-((x - 0.5)^2 + y^2)) - 0.1*exp(-((x + 0.5)^2 + y^2))
hᵢ(x, y, z) = 1
uᵢ(x, y, z) = 5*y*exp(-(x^2 + y^2))
vᵢ(x, y, z) = -5*x*exp(-(x^2 + y^2))
set!(model, u = uᵢ, v = vᵢ, h = hᵢ, A = Aᵢ)
simulation = Simulation(model, Δt = 0.01, stop_time = 30.0)


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
A = model.tracers.A
B_x = -∂y(A)/h
B_y = ∂x(A)/h
compute!(s)

kinetic_energy_func(args...) = mean((1/2)*h*(u^2 + v^2))*Lx*Ly
magnetic_energy_func(args...) = mean((1/2)*h*(B_x^2 + B_y^2))*Lx*Ly
potential_energy_func(args...) = mean((1/2)*model.gravitational_acceleration*((h-hᵢ)^2))*Lx*Ly
total_energy_func(args...) = mean((1/2)*h*(u^2 + v^2))*Lx*Ly + mean((1/2)*h*(B_x^2 + B_y^2))*Lx*Ly + mean((1/2)*model.gravitational_acceleration*((h-hᵢ)^2))*Lx*Ly


filename = "SW_MHD_adjustment"
simulation.output_writers[:fields] = JLD2OutputWriter(model, (; u, v, A = model.tracers.A, s),
                                                      schedule = TimeInterval(0.1),
                                                      filename = filename * ".jld2",
                                                      overwrite_existing = true)

energies_filename = joinpath(@__DIR__, "energies.nc")
simulation.output_writers[:energies] = NetCDFOutputWriter(model, (; kinetic_energy_func, magnetic_energy_func, potential_energy_func, total_energy_func),
                                                        filename = energies_filename,
                                                        array_type = Array{Float64},
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
ax_s = Axis(fig[1,3], xlabel = "x", ylabel = "y", title=title_s)
hm1 = heatmap!(ax_A, x, y, A, colormap=:deep, colorrange = (minimum(A_timeseries.data), maximum(A_timeseries.data)))
Colorbar(fig[1,2], hm1)
hm2 = heatmap!(ax_s, x, y, s, colormap=:deep, colorrange = (minimum(s_timeseries.data), maximum(s_timeseries.data)))
Colorbar(fig[1,4], hm2)

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


initial_total_energy = first(total_energy)
deviation_total_energy = abs.(total_energy .- initial_total_energy) .* 100

f = Figure()

Axis(f[1, 1], title = "kinetic energy")
lines!(t, kinetic_energy; linewidth = 4, color = "red")

Axis(f[1, 2], title = "magnetic energy")
lines!(t, magnetic_energy; linewidth = 4, color = "blue")

Axis(f[2, 1], title = "potential energy")
lines!(t, potential_energy; linewidth = 4, color = "green")

Axis(f[2, 2], title = "relative energy error (%)")
lines!(t, deviation_total_energy; linewidth = 4, color = "black")

Label(f[0, :], "64x64 Two_Gaussians_Low_B: Energy Plots", textsize = 20)

save("energy_plot.png", f)
