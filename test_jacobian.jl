using Oceananigans
using Oceananigans.Operators
#using PyPlot

Lx, Ly = 10,  10
Nx, Ny = 100, 100

grid = RectilinearGrid(CPU(); size = (Nx, Ny), 
                      x = (-Lx/2, Lx/2), y = (-Ly/2, Ly/2),
                      topology = (Periodic, Periodic, Flat))

xᶜ, xᶠ = grid.xᶜᵃᵃ, grid.xᶠᵃᵃ
yᶜ, yᶠ = grid.yᵃᶜᵃ, grid.yᵃᶠᵃ 

   A(i, j, k, grid, x, y) = exp(-x[i]^2 -y[j]^2)
∂A_x(i, j, k, grid, x, y) = -2 * x[i] * exp(-x[i]^2 -y[j]^2)
∂A_y(i, j, k, grid, x, y) = -2 * y[j] * exp(-x[i]^2 -y[j]^2)

∂A_xx(i, j, k, grid, x, y) = (4 * x[i]^2 - 2) * exp(-x[i]^2 -y[j]^2)
∂A_xy(i, j, k, grid, x, y) = 4 * x[i] * y[j] * exp(-x[i]^2 -y[j]^2)

∂A_yx(i, j, k, grid, x, y) = 4 * x[i] * y[j] * exp(-x[i]^2 -y[j]^2)
∂A_yy(i, j, k, grid, x, y) = (4 * y[j]^2 - 2) * exp(-x[i]^2 -y[j]^2)

 jacobian_x_exact = zeros(Float64, length(xᶠ), length(yᶠ))
jacobian_x_approx = zeros(Float64, length(xᶜ), length(yᶜ))

 jacobian_y_exact = zeros(Float64, length(xᶠ), length(yᶠ))
jacobian_y_approx = zeros(Float64, length(xᶜ), length(yᶜ))

for i = 1:Nx, j = 1:Ny
    jacobian_x_approx[i, j] = ∂xᶠᶜᶜ(i, j, 1, grid, A, xᶜ, yᶜ) * ℑxyᶠᶜᵃ(i, j, 1, grid, ∂yᶜᶠᶜ, ∂A_x, xᶜ, yᶜ)
                            - ∂xᶠᶜᶜ(i, j, 1, grid, ∂A_x, xᶜ, yᶜ) * ℑxyᶠᶜᵃ(i, j, 1, grid, ∂yᶜᶠᶜ, A, xᶜ, yᶜ)
    jacobian_y_approx[i, j] = ℑxyᶜᶠᵃ(i, j, 1, grid, ∂xᶠᶜᶜ, A, xᶜ, yᶜ) * ∂yᶜᶠᶜ(i, j, 1, grid, ∂A_y, xᶜ, yᶜ)
                            - ℑxyᶜᶠᵃ(i, j, 1, grid, ∂xᶠᶜᶜ, ∂A_y, xᶜ, yᶜ) * ∂yᶜᶠᶜ(i, j, 1, grid, A, xᶜ, yᶜ)
end

for i = 1:Nx, j = 1:Ny
    jacobian_x_exact[i, j] = ∂A_x(i, j, 1, grid, xᶠ, yᶜ) * ∂A_xy(i, j, 1, grid, xᶠ, yᶜ) 
                           - ∂A_xx(i, j, 1, grid, xᶠ, yᶜ) * ∂A_y(i, j, 1, grid, xᶠ, yᶜ)
    jacobian_y_exact[i, j] = ∂A_x(i, j, 1, grid, xᶜ, yᶠ) * ∂A_yy(i, j, 1, grid, xᶜ, yᶠ)
                           - ∂A_yx(i, j, 1, grid, xᶜ, yᶠ) * ∂A_y(i, j, 1, grid, xᶜ, yᶠ)
end


error_x = abs.(jacobian_x_exact .- jacobian_x_approx)
error_y = abs.(jacobian_y_exact .- jacobian_y_approx)

#plt_jacobian = plot_surface(xᶜ, yᶜ, jacobian_exact)
#plt_jacobian_approx = plot_surface(xᶜ, yᶜ, jacobian_approx)
#plt_error = plot_surface(xᶜ, yᶜ, error)


print("Maximum error =", maximum(abs, error_x), "\n")

print("Maximum error =", maximum(abs, error_y), "\n")