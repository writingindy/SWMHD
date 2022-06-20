using Oceananigans
using Oceananigans.Models.ShallowWaterModels: VectorInvariantFormulation
using Oceananigans.Advection: VelocityStencil, VorticityStencil
using Oceananigans.Operators: ℑxᶜᵃᵃ, ∂xᶠᶜᶜ, ℑyᵃᶜᵃ, ∂yᶜᶠᶜ, ℑxyᶠᶜᵃ, ℑxyᶜᶠᵃ, ℑxᶠᵃᵃ, ℑyᵃᶠᵃ
using CairoMakie, Statistics, JLD2, Printf

include("sw_mhd_jacobian_functions.jl")

Lx, Ly = 10, 10

grid = RectilinearGrid(size = (128, 128), 
                          x = (-Lx/2, Lx/2), y = (-Ly/2, Ly/2), 
                   topology = (Periodic, Periodic, Flat))

using Oceananigans.TurbulenceClosures
#horizontal_diffusivity = HorizontalScalarDiffusivity(ν=νh)
biharmonic_viscosity   = HorizontalScalarBiharmonicDiffusivity(ν=1e-3, κ=1e-3)

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
                          closure = biharmonic_viscosity,
                          formulation = VectorInvariantFormulation()
                          )

Aᵢ(x, y, z) = exp(-(x^2 + y^2))
set!(model, h = 1, A = Aᵢ)
simulation = Simulation(model, Δt = 0.001, stop_time = 10.0)

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
run!(simulation)


output_prefix = "SW_MHD_adjustment"
filepath = output_prefix * ".jld2"
file = jldopen(filepath)

x, y, z = nodes((Center, Center, Center), grid)

@info "Making a movie of the magnetic potential function A..."

iter = Observable(0)
iters = parse.(Int, keys(file["timeseries/t"]))
A = @lift(file["timeseries/A/" * string($iter)][:, :, 1])
s = @lift(file["timeseries/s/" * string($iter)][:, :, 1])
title_A = @lift(@sprintf("Magnetic potential at time = %s", prettytime(file["timeseries/t/" * string($iter)])))
title_s = @lift(@sprintf("Speed at time = %s", prettytime(file["timeseries/t/" * string($iter)])))
fig = Figure(resolution = (800, 400))
ax_A = Axis(fig[1,1], xlabel = "x", ylabel = "y", title=title_A)
ax_s = Axis(fig[1,2], xlabel = "x", ylabel = "y", title=title_s)
heatmap_A = heatmap!(ax_A, x, y, A, colormap=:deep)
heatmap_s = heatmap!(ax_s, x, y, s, colormap=:deep)
display(fig)

record(fig, output_prefix * ".mp4", iters[2:end], framerate=6) do i
    @info "Plotting iteration $i of $(iters[end])..."
    iter[] = i
end