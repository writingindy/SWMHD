using Oceananigans
using Oceananigans.Models.ShallowWaterModels: ConservativeFormulation
using Oceananigans.Advection: VelocityStencil, VorticityStencil
using Oceananigans.Operators: ℑxᶜᵃᵃ, ∂xᶠᶜᶜ, ℑyᵃᶜᵃ, ∂yᶜᶠᶜ, ℑxyᶠᶜᵃ, ℑxyᶜᶠᵃ, ℑxᶠᵃᵃ, ℑyᵃᶠᵃ, δxᶜᵃᵃ, δxᶠᵃᵃ, δyᵃᶠᵃ, δyᵃᶜᵃ, Vᶠᶜᶜ, Vᶜᶠᶜ
using CairoMakie, Statistics, JLD2, Printf


using Oceananigans.Advection: 
    _advective_momentum_flux_Uu,
    _advective_momentum_flux_Uv,
    _advective_momentum_flux_Vu,
    _advective_momentum_flux_Vv

using Oceananigans.Grids: AbstractGrid
using Oceananigans.Operators: Ax_qᶠᶜᶜ, Ay_qᶜᶠᶜ

include("new_functions.jl")

Lx, Ly = 10, 10

grid = RectilinearGrid(size = (64, 64), 
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

Aᵢ(x, y, z) = 0.1exp(-((x - 0.5)^2 + y^2)) - 0.5exp(-((x + 0.5)^2 + y^2))

#uᵢ(x, y, z) = 5y*exp(-(x^2 + y^2))
#vᵢ(x, y, z) = -5x*exp(-(x^2 + y^2))
set!(model#=, u = uᵢ, v = vᵢ=#, h = 1, A = Aᵢ)
simulation = Simulation(model, Δt = 0.01, stop_time = 35.0)

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
compute!(s)

filename = "SW_MHD_adjustment"

simulation.output_writers[:fields] = JLD2OutputWriter(model, (; u, v, A = model.tracers.A, s),
                                                      schedule = TimeInterval(0.1),
                                                      filename = filename * ".jld2",
                                                      overwrite_existing = true)

@info "Running with Δt = $(prettytime(simulation.Δt))"
run!(simulation)
