using Oceananigans
using Oceananigans.Operators
using PyPlot

Lx = 10
Ly = 10
Nx = 400
Ny = 400

test_grid = RectilinearGrid(CPU(); size = (Nx, Ny), x = (-Lx/2, Lx/2), y = (-Ly/2, Ly/2),
                      topology = (Periodic, Periodic, Flat))

xᶜ = test_grid.xᶜᵃᵃ
xᶠ = test_grid.xᶠᵃᵃ

yᶜ = test_grid.yᵃᶜᵃ
yᶠ = test_grid.yᵃᶠᵃ

f(i, j, k, grid, x, y) = exp(-x[i]^2 -y[j]^2)
∂f_x(i, j, k, grid, x, y) = -2 * x[i] * exp(-x[i]^2 -y[j]^2)
∂f_y(i, j, k, grid, x, y) = -2 * y[j] * exp(-x[i]^2 -y[j]^2)

g(i, j, k, grid, x, y) = sin(x[i]) * cos(y[j])
∂g_x(i, j, k, grid, x, y) = cos(x[i])* cos(y[j])
∂g_y(i, j, k, grid, x, y) = -sin(x[i]) * sin(y[j])

jacobian_func(i, j, k, grid, x, y) =  ∂f_x(i, j, k, grid, x, y)#= * ∂g_y(i, j, k, grid, x, y) =# #= - (∂f_y(i, j, k, grid, x, y) * ∂g_x(i, j, k, grid, x, y))=#

fᶜ = zeros(Float64, length(xᶜ), length(yᶜ))
∂f_xᶠ = zeros(Float64, length(xᶠ), length(yᶠ))
∂f_yᶠ = zeros(Float64, length(xᶠ), length(yᶠ))

gᶜ = zeros(Float64, length(xᶜ), length(yᶜ))
∂g_xᶠ = zeros(Float64, length(xᶠ), length(yᶠ))
∂g_yᶠ = zeros(Float64, length(xᶠ), length(yᶠ))

jacobian = zeros(Float64, length(xᶠ), length(yᶠ))

jacobian_approx = zeros(Float64, length(xᶜ), length(yᶜ))

# Are xᶜ and xᶠ (similarly for y) ever arrays with different lengths?
for (index, i) in enumerate(1:Nx)
    for (index, j) in enumerate(1:Ny)
        fᶜ[i, j] = f(i, j, 1, test_grid, xᶜ, yᶜ)
        gᶜ[i, j] = g(i, j, 1, test_grid, xᶜ, yᶜ)
        jacobian_approx[i, j] =   ∂xᶠᶜᶜ(i, j, 1, test_grid, f, xᶜ, yᶜ) * ∂yᶜᶠᶜ(i, j, 1, test_grid, g,  xᶜ, yᶜ)  - ∂xᶠᶜᶜ(i, j, 1, test_grid, g, xᶜ, yᶜ) * ∂yᶜᶠᶜ(i, j, 1, test_grid, f, xᶜ, yᶜ)
    end
end

for (index, i) in enumerate(1:Nx)
    for (index, j) in enumerate(1:Ny)
        ∂f_xᶠ[i, j] = ∂f_x(i, j, 1, test_grid, xᶠ, yᶜ)
        ∂f_yᶠ[i, j] = ∂f_y(i, j, 1, test_grid, xᶜ, yᶠ)
        ∂g_xᶠ[i, j] = ∂g_x(i, j, 1, test_grid, xᶠ, yᶜ)
        ∂g_yᶠ[i, j] = ∂g_y(i, j, 1, test_grid, xᶜ, yᶠ)
        jacobian[i, j] = ∂f_xᶠ[i, j] * ∂g_yᶠ[i, j] - ∂f_yᶠ[i, j] * ∂g_xᶠ[i, j]
    end
end

#∂f_xᶠ[i] = ∂f_x(i, 1, k, grid, xᶠ , yᶠ)
#∂f_x_grid = [∂f_x for ∂f_x in ∂f_xᶠ]


error = abs.(jacobian .- jacobian_approx)


#plt_f = plot_surface(xᶜ, yᶜ, fᶜ);
#plt_g = plot_surface(xᶜ[1:Nx], yᶜ[1:Nx], gᶜ[1:(end-6), 1:(end-6)])
plt_jacobian = plot_surface(xᶜ, yᶜ, jacobian)
plt_jacobian_approx = plot_surface(xᶜ, yᶜ, jacobian_approx)
#plt_error = plot_surface(, y_grid[2:end], error)


print("Maximum error =", maximum(abs, error), "\n")