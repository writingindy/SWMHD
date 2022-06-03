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

exp_var = 0.2

   A(i, j, k, grid, x, y) = exp(-exp_var * (x[i])^2 - exp_var * (y[j])^2)
∂A_x(i, j, k, grid, x, y) = -(exp_var * 2) * x[i] * exp(-exp_var * (x[i])^2 - exp_var * (y[j])^2)
∂A_y(i, j, k, grid, x, y) = -(exp_var * 2) * y[j] * exp(-exp_var * (x[i])^2 - exp_var * (y[j])^2)

jacobian_x_approx = zeros(Float64, length(xᶜ), length(yᶜ))
jacobian_y_approx = zeros(Float64, length(xᶜ), length(yᶜ))

∇A_x =  zeros(Float64, length(xᶜ), length(yᶜ))
∇A_y = zeros(Float64, length(xᶜ), length(yᶜ))

A_num = zeros(Float64, length(xᶜ), length(yᶜ))

for i = 1:Nx, j = 1:Ny
    A_num[i, j] = A(i, j, 1, grid, xᶜ, yᶜ)
    ∇A_x[i, j] = ∂xᶠᶜᶜ(i, j, 1, grid, A, xᶜ, yᶜ)
    ∇A_y[i, j] = ∂yᶜᶠᶜ(i, j, 1, grid, A, xᶜ, yᶜ)

    jacobian_x_approx[i, j] = ∂xᶠᶜᶜ(i, j, 1, grid, A, xᶜ, yᶜ) * ℑxyᶠᶜᵃ(i, j, 1, grid, ∂yᶜᶠᶜ, ∂A_x, xᶜ, yᶜ)
                            - ∂xᶠᶜᶜ(i, j, 1, grid, ∂A_x, xᶜ, yᶜ) * ℑxyᶠᶜᵃ(i, j, 1, grid, ∂yᶜᶠᶜ, A, xᶜ, yᶜ)
    jacobian_y_approx[i, j] = ℑxyᶜᶠᵃ(i, j, 1, grid, ∂xᶠᶜᶜ, A, xᶜ, yᶜ) * ∂yᶜᶠᶜ(i, j, 1, grid, ∂A_y, xᶜ, yᶜ)
                            - ℑxyᶜᶠᵃ(i, j, 1, grid, ∂xᶠᶜᶜ, ∂A_y, xᶜ, yᶜ) * ∂yᶜᶠᶜ(i, j, 1, grid, A, xᶜ, yᶜ)
end

lorentz_force_x = jacobian_y_approx
lorentz_force_y = -1 .* jacobian_x_approx

B_x = -1 .* ∇A_y
B_y = ∇A_x

#plt = plot_surface(xᶜ, yᶜ, A_num, cmap=ColorMap("coolwarm"))

step = 10

plt_A = colorbar(contourf(xᶜ, yᶜ, A_num))
plt_lorentz = quiver(xᶜ[1:step:end], yᶜ[1:step:end], lorentz_force_x[1:step:end, 1:step:end], lorentz_force_y[1:step:end, 1:step:end], scale = 4)
title("Contour plot of A and quiver plot of Lorentz force")
xlim((-4, 4))
ylim((-4, 4))

#=plt_A = colorbar(contourf(xᶜ, yᶜ, A_num))
plt_B = quiver(xᶜ[1:step:end], yᶜ[1:step:end], B_x[1:step:end, 1:step:end], B_y[1:step:end, 1:step:end], scale = 15)
title("Contour plot of A and quiver plot of magnetic field")
xlim((-4, 4))
ylim((-4, 4))=#