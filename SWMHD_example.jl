using Oceananigans
using Oceananigans.Models.ShallowWaterModels: VectorInvariantFormulation
using Oceananigans.Advection: VelocityStencil, VorticityStencil


Lx, Ly, Lz = 2π, 20, 1
Nx, Ny = 128, 128

grid = RectilinearGrid(size = (Nx, Ny), x = (0, Lx), y = (-Ly/2, Ly/2), topology = (Periodic, Bounded, Flat))

u_forcing_func(x, y, t, A, h) = (1/h)*(∂x(A)*∂y(∂y(A)/h) - ∂y(A)*∂x(∂y(A)/h))
                  
v_forcing_func(x, y, t, A, h) = (1/h)*(∂x(A)*∂y(∂x(A)/h) - ∂y(A)*∂x(∂x(A)/h))

u_forcing = Forcing(u_forcing_func, field_dependencies = (:A, :h))

v_forcing = Forcing(v_forcing_func, field_dependencies = (:A, :h))

const U = 1.0         # Maximum jet velocity

f = 1         # Coriolis parameter
g = 9.8         # Gravitational acceleration
Δη = f * U / g  # Maximum free-surface deformation as dictated by geostrophy


model = ShallowWaterModel(
                          timestepper = :RungeKutta3,
                          momentum_advection = WENO5(vector_invariant = VelocityStencil()),
                          grid = grid,
                          gravitational_acceleration = g,
                          coriolis = FPlane(f=f),
                          tracers = (:A),
                          forcing = (u = u_forcing, v = v_forcing),
                          formulation = VectorInvariantFormulation())

h̄(x, y, z) = Lz - Δη * tanh(y)
ū(x, y, z) = U * sech(y)^2

ω̄(x, y, z) = 2 * U * sech(y)^2 * tanh(y)

small_amplitude = 1e-4

uⁱ(x, y, z) = ū(x, y, z) + small_amplitude * exp(-y^2) * randn()

A_i(x, y, z) = -y + randn()

set!(model, u = ū, h = h̄, A = A_i)

u, v, h = model.solution

# Build and compute mean vorticity discretely
ω = Field(∂x(v) - ∂y(u))
compute!(ω)

# Copy mean vorticity to a new field
ωⁱ = Field{Face, Face, Nothing}(model.grid)
ωⁱ .= ω

# Use this new field to compute the perturbation vorticity
ω′ = Field(ω - ωⁱ)

set!(model, u = uⁱ)

simulation = Simulation(model, Δt = 1e-2, stop_time = 150)

using LinearAlgebra: norm

perturbation_norm(args...) = norm(v)

simulation.output_writers[:fields] = NetCDFOutputWriter(model, (; ω, ω′),
                                                        filename = joinpath(@__DIR__, "shallow_water_Bickley_jet_fields.nc"),
                                                        schedule = TimeInterval(1),
                                                        overwrite_existing = true)

simulation.output_writers[:growth] = NetCDFOutputWriter(model, (; perturbation_norm),
                                                        filename = joinpath(@__DIR__, "shallow_water_Bickley_jet_perturbation_norm.nc"),
                                                        schedule = IterationInterval(1),
                                                        dimensions = (; perturbation_norm = ()),
                                                        overwrite_existing = true)

run!(simulation)


using NCDatasets, Plots, Printf


x, y = xnodes(ω), ynodes(ω)

kwargs = (
         xlabel = "x",
         ylabel = "y",
         aspect = 1,
           fill = true,
         levels = 20,
      linewidth = 0,
          color = :balance,
       colorbar = true,
           ylim = (-Ly/2, Ly/2),
           xlim = (0, Lx)
)

ds = NCDataset(simulation.output_writers[:fields].filepath, "r")

anim = @animate for (iter, t) in enumerate(ds["time"])
    ω = ds["ω"][:, :, 1, iter]
    ω′ = ds["ω′"][:, :, 1, iter]

    ω′_max = maximum(abs, ω′)

    plot_ω = contour(x, y, ω',
                     clim = (-1, 1),
                     title = @sprintf("Total vorticity, ω, at t = %.1f", t); kwargs...)

    plot_ω′ = contour(x, y, ω′',
                      clim = (-ω′_max, ω′_max),
                      title = @sprintf("Perturbation vorticity, ω - ω̄, at t = %.1f", t); kwargs...)

    plot(plot_ω, plot_ω′, layout = (1, 2), size = (800, 440))
end

close(ds)

mp4(anim, "shallow_water_Bickley_jet.mp4", fps=15)