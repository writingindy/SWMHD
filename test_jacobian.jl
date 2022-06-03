using Oceananigans
using Oceananigans.Operators
using PyPlot

Lx, Ly = 10,  10
Nx, Ny = 400, 400

grid = RectilinearGrid(CPU(); size = (Nx, Ny), 
                      x = (-Lx/2, Lx/2), y = (-Ly/2, Ly/2),
                      topology = (Periodic, Periodic, Flat))

xᶜ, xᶠ = grid.xᶜᵃᵃ, grid.xᶠᵃᵃ
yᶜ, yᶠ = grid.yᵃᶜᵃ, grid.yᵃᶠᵃ 

   f(i, j, k, grid, x, y) = exp(-x[i]^2 -y[j]^2)
∂f_x(i, j, k, grid, x, y) = -2 * x[i] * exp(-x[i]^2 -y[j]^2)
∂f_y(i, j, k, grid, x, y) = -2 * y[j] * exp(-x[i]^2 -y[j]^2)

   g(i, j, k, grid, x, y) = sin(x[i]) * cos(y[j])
∂g_x(i, j, k, grid, x, y) = cos(x[i])* cos(y[j])
∂g_y(i, j, k, grid, x, y) = -sin(x[i]) * sin(y[j])

∂f_xᶠ = zeros(Float64, length(xᶠ), length(yᶠ))
∂f_yᶠ = zeros(Float64, length(xᶠ), length(yᶠ))
∂g_xᶠ = zeros(Float64, length(xᶠ), length(yᶠ))
∂g_yᶠ = zeros(Float64, length(xᶠ), length(yᶠ))

 jacobian_exact = zeros(Float64, length(xᶠ), length(yᶠ))
jacobian_approx = zeros(Float64, length(xᶜ), length(yᶜ))

for i = 1:Nx, j = 1:Ny
    jacobian_approx[i, j] = ∂xᶠᶜᶜ(i, j, 1, grid, f, xᶜ, yᶜ) * ∂yᶜᶠᶜ(i, j, 1, grid, g,  xᶜ, yᶜ)  
                          - ∂xᶠᶜᶜ(i, j, 1, grid, g, xᶜ, yᶜ) * ∂yᶜᶠᶜ(i, j, 1, grid, f, xᶜ, yᶜ)
end

for i = 1:Nx, j = 1:Ny
    jacobian_exact[i, j] = ∂f_x(i, j, 1, grid, xᶠ, yᶜ) * ∂g_y(i, j, 1, grid, xᶜ, yᶠ) 
                         - ∂f_y(i, j, 1, grid, xᶜ, yᶠ) * ∂g_x(i, j, 1, grid, xᶠ, yᶜ)
end

error = abs.(jacobian_exact .- jacobian_approx)

plt_jacobian = plot_surface(xᶜ, yᶜ, jacobian)
plt_jacobian_approx = plot_surface(xᶜ, yᶜ, jacobian_approx)
#plt_error = plot_surface(, y_grid[2:end], error)


print("Maximum error =", maximum(abs, error), "\n")
