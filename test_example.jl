using Oceananigans
using Oceananigans.Models.ShallowWaterModels: VectorInvariantFormulation
using Oceananigans.Advection: VelocityStencil, VorticityStencil
using Oceananigans.Operators: ℑxᶜᵃᵃ, ∂xᶠᶜᶜ, ℑyᵃᶜᵃ, ∂yᶜᶠᶜ, ℑxyᶠᶜᵃ, ℑxyᶜᶠᵃ, ℑxᶠᵃᵃ, ℑyᵃᶠᵃ
using CairoMakie, Statistics

# Computes ∂x(A)/h at ccc
# A and h are at centers, ccc
# ∂x(A) is at fcc, and we interpolate it to ccc
function ∂xA_over_h(i, j, k, grid, A, h)
    return ℑxᶜᵃᵃ(i, j, k, grid, ∂xᶠᶜᶜ, A)/h[i, j, k]
end

# Computes ∂y(A)/h at ccc
# A and h are at centers, ccc
# ∂y(A) is at cfc, and we interpolate it to ccc
function ∂yA_over_h(i, j, k, grid, A, h)
    return ℑyᵃᶜᵃ(i, j, k, grid, ∂yᶜᶠᶜ, A)/h[i, j, k]
end

# I added these functions because I thought that maybe the issue was with how the
# interpolation operator was working but these are probably redundant and can go
# back into the jacobian functions

function ∂yyA(i, j, k, grid, A, h)
    return ∂yᶜᶠᶜ(i, j, k, grid, ∂yA_over_h, A, h)
end

function ∂xxA(i, j, k, grid, A, h)
    return ∂xᶠᶜᶜ(i, j, k, grid, ∂xA_over_h, A, h)
end

# I also tried inline functions, read through the documentation for the derivative
# operators and interpolation operators and I'm pretty sure the syntax for all of
# this is correct

# Computes the Jacobian at fcc for the u-component forcing term
function jacobian_x(i, j, k, grid, fields)
    return (∂xᶠᶜᶜ(i, j, k, grid, fields.A) * ℑxyᶠᶜᵃ(i, j, k, grid, ∂yyA, fields.A, fields.h)) - (ℑxyᶠᶜᵃ(i, j, k, grid, ∂yᶜᶠᶜ, fields.A) * ∂xᶠᶜᶜ(i, j, k, grid, ∂yA_over_h, fields.A, fields.h))
end


# Computes the Jacobian at cfc for the v-component forcing term
function jacobian_y(i, j, k, grid, fields)
    return (ℑxyᶜᶠᵃ(i, j, k, grid, ∂xᶠᶜᶜ, fields.A) * ∂yᶜᶠᶜ(i, j, k, grid, ∂xA_over_h, fields.A, fields.h)) - (∂yᶜᶠᶜ(i, j, k, grid, fields.A) * ℑxyᶜᶠᵃ(i, j, k, grid, ∂xxA, fields.A, fields.h))
end

# Computes the u-component forcing term at fcc 
# Note that jacobian_y() is used because -ẑ × ŷ = x̂
function Lorentz_forcing_term_x(i, j, k, grid, clock, fields)
    return (1/ℑxᶠᵃᵃ(i, j, k, grid, fields.h))*(jacobian_x(i, j, k, grid, fields))
end

# Computes the v-component forcing term at cfc; 
# Note that jacobian_x() is used because -ẑ × x̂ = -ŷ
function Lorentz_forcing_term_y(i, j, k, grid, clock, fields)
    return (-1/ℑyᵃᶠᵃ(i, j, k, grid, fields.h))*(jacobian_y(i, j, k, grid, fields))
end


# Model parameters

Lx, Ly, Lz = 2π, 20, 1
Nx, Ny = 128, 128

const U = 1.0         # Maximum jet velocity

f = 1         # Coriolis parameter
g = 9.81         # Gravitational acceleration
Δη = f * U / g  # Maximum free-surface deformation as dictated by geostrophy

coriolis = FPlane(f=1)

# I changed the x-limits to be symmetric because the gaussian bump is centered at zero
grid = RectilinearGrid(size = (Nx, Ny), x = (-Lx/2, Lx/2), y = (-Ly/2, Ly/2), topology = (Periodic, Periodic, Flat))

## Forcing functions for the SWMHD model

Lorentz_force_x = Forcing(Lorentz_forcing_term_x, discrete_form = true)
Lorentz_force_y = Forcing(Lorentz_forcing_term_y, discrete_form = true)

## Construction of SWMHD model

model = ShallowWaterModel(
                          timestepper = :RungeKutta3,
                          momentum_advection = WENO5(vector_invariant = VelocityStencil()),
                          grid = grid,
                          gravitational_acceleration = g,
                          coriolis = FPlane(f = 1),
                          tracers = (:A),
                          forcing = (u = Lorentz_force_x, v = Lorentz_force_y),
                          formulation = VectorInvariantFormulation())

## Background state and perturbation

# I changed the initial height to be constant everywhere because for initial conditions
# where the height is constant, u-velocity and v-velocity are zero, for tracer A and 
# no forcing we expect the velocities to stay the same (at zero), and the tracer to 
# stay the same as well which was what I got - if the height isn't constant, it 
# causes motion in the x direction 
h̄(x, y, z) = 1
ū(x, y, z) = 0
ω̄(x, y, z) = 0#2 * U * sech(y)^2 * tanh(y)

small_amplitude = 1e-4


# When I comment out the randn(), for small enough time steps I can kind of see a ripple
Aᵢ(x, y, z) = exp(-(x^2 + y^2)) #+ 0.01*randn()

u, v, h = model.solution
# zero initial velocities
uᵢ = zeros(size(u)...)
vᵢ = zeros(size(v)...)

set!(model, u = uᵢ, v = vᵢ, h = h̄, A = Aᵢ)

# I haven't tried time steps smaller than 0.0001 yet
simulation = Simulation(model, Δt = 0.0001, stop_time = 10)

u, v, h = model.solution

ω = ∂x(v) - ∂y(u)

s = sqrt(u^2 + v^2)

filename = "two_dimensional_turbulence"

# need to remember to change TimeInterval as time step changes
simulation.output_writers[:fields] = JLD2OutputWriter(model, (; ω, s, u = model.solution.u, v = model.solution.v, A = model.tracers.A),
                                                      schedule = TimeInterval(0.0006),
                                                      filename = filename * ".jld2",
                                                      overwrite_existing = true)


run!(simulation)

# I copied and adapted code from the "2D Turbulence" example for visualizing things

ω_timeseries = FieldTimeSeries(filename * ".jld2", "ω")
s_timeseries = FieldTimeSeries(filename * ".jld2", "s")
A_timeseries = FieldTimeSeries(filename * ".jld2", "A")
u_timeseries = FieldTimeSeries(filename * ".jld2", "u")
v_timeseries = FieldTimeSeries(filename * ".jld2", "v")


times = ω_timeseries.times

xω, yω, zω = nodes(ω_timeseries)
xs, ys, zs = nodes(s_timeseries)
xA, yA, zA = nodes(A_timeseries)
xu, yu, zu = nodes(u_timeseries)
xv, yv, zv = nodes(v_timeseries)


set_theme!(Theme(fontsize = 24))

@info "Making a neat movie of vorticity and speed..."

fig = Figure(resolution = (800, 500))

axis_kwargs = (xlabel = "x",
               ylabel = "y",
               limits = ((-π, π), (-π, π)),
               aspect = AxisAspect(1))

#ax_ω = Axis(fig[2, 1]; title = "Vorticity", axis_kwargs...)
ax_s = Axis(fig[2, 1]; title = "Speed", axis_kwargs...)
ax_A = Axis(fig[2, 2]; title = "Stream Function", axis_kwargs...)
ax_velocity = Axis(fig[2, 3]; title = "Velocity", axis_kwargs...)


n = Observable(1)

#ω = @lift interior(ω_timeseries[$n], :, :, 1)
s = @lift interior(s_timeseries[$n], :, :, 1)
A = @lift interior(A_timeseries[$n], :, :, 1)
u = @lift interior(u_timeseries[$n], :, :, 1)
v = @lift interior(v_timeseries[$n], :, :, 1)

#colorbar(heatmap!(ax_ω, xω, yω, ω; colormap = :balance, colorrange = (-2, 2)))
heatmap!(ax_s, xs, ys, s; colormap = :speed, colorrange = (0, 0.2))
# Tried to animate contour plots but for some reason it wasn't working so I just
# plotted a heatmap instead which works
heatmap!(ax_A ,xA, yA, A; colormap = :speed, colorrange = (0, 2))


# Sometimes the quiver plot doesn't generate properly so I commented it out
# Not very interesting behaviour anyways

#step = 5
#quiver!(xu[1:step:128], yu[1:step:128], u, v)



title = @lift "t = " * string(round(times[$n], digits=2))
Label(fig[1, 1:2], title, textsize=24, tellwidth=false)

frames = 1:length(times)

@info "Making a neat animation of vorticity and speed..."

record(fig, filename * ".mp4", frames, framerate=24) do i
    msg = string("Plotting frame ", i, " of ", frames[end])
    print(msg * " \r")
    n[] = i
end
