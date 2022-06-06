using Oceananigans
using Oceananigans.Operators
using PyPlot
using LinearAlgebra

Lx, Ly = 10,  10
Nx, Ny = 400, 400

ℓ = 2
c = -1

grid = RectilinearGrid(CPU(); size = (Nx, Ny), 
                      x = (-Lx/2, Lx/2), y = (-Ly/2, Ly/2),
                      topology = (Periodic, Periodic, Flat))

xᶜ, xᶠ = grid.xᶜᵃᵃ, grid.xᶠᵃᵃ
yᶜ, yᶠ = grid.yᵃᶜᵃ, grid.yᵃᶠᵃ


   A(i, j, k, grid, x, y) = c* exp( - (x[i]^2 + y[j]^2) / ℓ^2)
∂A_x(i, j, k, grid, x, y) = -c * (2 / ℓ^2) * x[i] * exp( - (x[i]^2 + y[j]^2) / ℓ^2)
∂A_y(i, j, k, grid, x, y) = -c * (2 / ℓ^2) * y[j] * exp( - (x[i]^2 + y[j]^2) / ℓ^2)

∂A_xx(i, j, k, grid, x, y) = c * ((4 * x[i]^2 - 2*ℓ^2) / ℓ^4) * exp( - (x[i]^2 + y[j]^2) / ℓ^2)
∂A_xy(i, j, k, grid, x, y) = c * ((4 * x[i] * y[j]) / ℓ^4) * exp( - (x[i]^2 + y[j]^2) / ℓ^2)

∂A_yx(i, j, k, grid, x, y) = c * ((4 * x[i] * y[j]) / ℓ^4) * exp( - (x[i]^2 + y[j]^2) / ℓ^2)
∂A_yy(i, j, k, grid, x, y) = c * ((4 * y[j]^2 - 2*ℓ^2) / ℓ^4) * exp( - (x[i]^2 + y[j]^2) / ℓ^2)

∂A_x_num(i, j, k, grid, A, x, y) = ℑxᶜᵃᵃ(i, j, k, grid, ∂xᶠᶜᶜ, A, x, y)
∂A_y_num(i, j, k, grid, A, x, y) = ℑyᵃᶜᵃ(i, j, k, grid, ∂yᶜᶠᶜ, A, x, y)

Bx_num = zeros(Float64, length(xᶜ), length(yᶜ))
By_num = zeros(Float64, length(xᶜ), length(yᶜ))

A_num = zeros(Float64, length(xᶜ), length(yᶜ))

lorentz_x_num  = zeros(Float64, length(xᶜ), length(yᶜ))
lorentz_y_num  = zeros(Float64, length(xᶜ), length(yᶜ))

Bx = zeros(Float64, length(xᶜ), length(yᶜ))
By = zeros(Float64, length(xᶜ), length(yᶜ))

lorentz_x  = zeros(Float64, length(xᶜ), length(yᶜ))
lorentz_y = zeros(Float64, length(xᶜ), length(yᶜ))

for i = 1:Nx, j = 1:Ny
   A_num[i, j] = A(i, j, 1, grid, xᶜ, yᶜ)
   Bx_num[i, j] = -∂yᶜᶠᶜ(i, j, 1, grid, A, xᶜ, yᶜ)
   By_num[i, j] = ∂xᶠᶜᶜ(i, j, 1, grid, A, xᶜ, yᶜ)

   lorentz_x_num[i, j] = ((∂xᶠᶜᶜ(i, j, 1, grid, A, xᶜ, yᶜ) * ℑxyᶠᶜᵃ(i, j, 1, grid, ∂yᶜᶠᶜ, ∂A_y_num, A, xᶜ, yᶜ)) 
                       - (ℑxyᶠᶜᵃ(i, j, 1, grid, ∂yᶜᶠᶜ, A, xᶜ, yᶜ) * ∂xᶠᶜᶜ(i, j, 1, grid, ∂A_y_num, A, xᶜ, yᶜ)))
   lorentz_y_num[i, j] = ((∂yᶜᶠᶜ(i, j, 1, grid, A, xᶜ, yᶜ) * ℑxyᶜᶠᵃ(i, j, 1, grid, ∂xᶠᶜᶜ, ∂A_x_num, A, xᶜ, yᶜ)) 
                       - (∂yᶜᶠᶜ(i, j, 1, grid, ∂A_x_num, A, xᶜ, yᶜ) * ℑxyᶜᶠᵃ(i, j, 1, grid, ∂xᶠᶜᶜ, A, xᶜ, yᶜ)))

   Bx[i, j] = -∂A_y(i, j, 1, grid, xᶜ, yᶠ)
   By[i, j] = ∂A_x(i, j, 1, grid, xᶠ, yᶜ)

   lorentz_x[i, j] = ((∂A_x(i, j, 1, grid, xᶠ, yᶜ) * ∂A_yy(i, j, 1, grid, xᶠ, yᶜ)) 
                   - (∂A_y(i, j, 1, grid, xᶠ, yᶜ) * ∂A_xy(i, j, 1, grid, xᶠ, yᶜ)))
   lorentz_y[i, j] = ((∂A_y(i, j, 1, grid, xᶜ, yᶠ) * ∂A_xx(i, j, 1, grid, xᶜ, yᶠ)) 
                   - (∂A_x(i, j, 1, grid, xᶜ, yᶠ) * ∂A_yx(i, j, 1, grid, xᶜ, yᶠ)))
end

#plt = plot_surface(xᶜ, yᶜ, A_num, cmap=ColorMap("coolwarm"))

step = 10

#=
plt_A = colorbar(contourf(xᶜ, yᶜ, A_num))
plt_lorentz = quiver(xᶜ[1:step:end], yᶜ[1:step:end], lorentz_x_num[1:step:end, 1:step:end]', lorentz_y_num[1:step:end, 1:step:end]', scale = 3)
title("Contour plot of A and quiver plot of Lorentz force")
xlim((-4, 4))
ylim((-4, 4))
=#


plt_A = colorbar(contourf(xᶜ, yᶜ, A_num))
plt_B = quiver(xᶜ[1:step:end], yᶜ[1:step:end], Bx[1:step:end, 1:step:end]', By[1:step:end, 1:step:end]', scale = 15)
title("Contour plot of A and quiver plot of magnetic field")
xlim((-4, 4))
ylim((-4, 4))



error_Bx = abs.(Bx .- Bx_num)
error_By = abs.(By .- By_num)

error_lorentz_x = abs.(lorentz_x .- lorentz_x_num)
error_lorentz_y = abs.(lorentz_y .- lorentz_y_num)

print("Maximum error in Bx =", maximum(abs, error_Bx), "\n")
print("Maximum error in By =", maximum(abs, error_By), "\n")
print("Maximum error in lorentz_x =", maximum(abs, error_lorentz_x), "\n")
print("Maximum error in lorentz_y =", maximum(abs, error_lorentz_y), "\n")

