using Oceananigans
using Oceananigans.Models.ShallowWaterModels: ConservativeFormulation
using Oceananigans.Advection
using Oceananigans.Operators
using Oceananigans.Grids: AbstractGrid, topology
using CairoMakie, Statistics, JLD2, Printf, NCDatasets

Lx, Ly = 10, 10
Nx, Ny = 512, 512
grid = RectilinearGrid(; size = (Nx, Ny), 
                           x = (-Lx/2, Lx/2), y = (-Ly/2, Ly/2),
                   topology = (Periodic, Periodic, Flat))


filename = "divergence_two_Gaussians_high_B"
energies_filename = "divergence_two_Gaussians_high_B_energies"
output_prefix = "divergence_two_Gaussians_high_B"

x, y, z = nodes((Center, Center, Center), grid)
s_timeseries = FieldTimeSeries(filename * ".jld2", "s")
A_timeseries = FieldTimeSeries(filename * ".jld2", "A")
times = s_timeseries.times

@info "Making a movie of the magnetic potential function A and speed..."

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
hm2 = heatmap!(ax_s, x, y, s, colormap=:curl, colorrange = (minimum(s_timeseries.data), maximum(s_timeseries.data)))
Colorbar(fig[1,4], hm2)

frames = 2:length(times)

record(fig, output_prefix * ".mp4", frames, framerate=96) do i
    @info "Plotting iteration $i of $(frames[end])..."
    iter[] = i
end

@info "Making a plot of the various energies of the system..."

Bx_timeseries = FieldTimeSeries(filename * ".jld2", "B_x")
By_timeseries = FieldTimeSeries(filename * ".jld2", "B_y")
h_timeseries = FieldTimeSeries(filename * ".jld2", "h")

KE_timeseries = FieldTimeSeries(energies_filename * ".jld2", "kinetic_energy_func")
PE_timeseries = FieldTimeSeries(energies_filename * ".jld2", "potential_energy_func")

t = KE_timeseries.times

B_x = interior(Bx_timeseries)[:, :, 1, :]
B_y = interior(By_timeseries)[:, :, 1, :]
h = interior(h_timeseries)[:, :, 1, :]

magnetic_energy = zeros(length(t))
total_energy = zeros(length(t))

kinetic_energy = interior(KE_timeseries)[1, 1, 1, :]
potential_energy = interior(PE_timeseries)[1, 1, 1, :]

for i in 1:length(t)
    magnetic_energy[i] = (1/2)*mean(h[:, :, i])*(mean((B_x[:, :, i]).^2) + mean((B_y[:, :, i]).^2))*Lx*Ly
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

Axis(f[2, 1], title = "available potential energy")
lines!(t, potential_energy; linewidth = 4, color = "green")

Axis(f[2, 2], title = "relative energy error (%)")
lines!(t, deviation_total_energy; linewidth = 4, color = "black")

Label(f[0, :], "512x512 Two_Gaussians_High_B: Energy Plots", textsize = 20)

save("divergence_two_Gaussians_high_B_energy_plot.png", f)