using Oceananigans
using Oceananigans.Operators
using PyPlot

Lx, Ly = 10,  10
Nx, Ny = 100, 100

grid = RectilinearGrid(CPU(); size = (Nx, Ny), 
                      x = (-Lx/2, Lx/2), y = (-Ly/2, Ly/2),
                      topology = (Periodic, Periodic, Flat))

xᶜ, xᶠ = grid.xᶜᵃᵃ, grid.xᶠᵃᵃ
yᶜ, yᶠ = grid.yᵃᶜᵃ, grid.yᵃᶠᵃ 

exp_var = 0.2

   A(i, j, k, grid, x, y) = exp(-exp_var * (x[i])^2 -exp_var * (y[j])^2)
∂A_x(i, j, k, grid, x, y) = -(exp_var * 2) * x[i] * exp(-x[i]^2 -y[j]^2)
∂A_y(i, j, k, grid, x, y) = -(exp_var * 2) * y[j] * exp(-x[i]^2 -y[j]^2)

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

#plt_A = colorbar(contourf(xᶜ, yᶜ, A_num))
#plt_lorentz = quiver(xᶜ, yᶜ, lorentz_force_x, lorentz_force_y, scale = 5)
#title("Contour plot of A and quiver plot of Lorentz force")

plt_A = colorbar(contourf(xᶜ, yᶜ, A_num))
plt_B = quiver(xᶜ, yᶜ, B_x, B_y, scale = 50)
title("Contour plot of A and quiver plot of magnetic field")