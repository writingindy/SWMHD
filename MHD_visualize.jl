using Oceananigans
using Oceananigans.Operators
using PyPlot

Lx, Ly = 10,  10
Nx, Ny = 400, 400

ℓ = 2 

grid = RectilinearGrid(CPU(); size = (Nx, Ny), 
                      x = (-Lx/2, Lx/2), y = (-Ly/2, Ly/2),
                      topology = (Periodic, Periodic, Flat))

xᶜ, xᶠ = grid.xᶜᵃᵃ, grid.xᶠᵃᵃ
yᶜ, yᶠ = grid.yᵃᶜᵃ, grid.yᵃᶠᵃ


  A(i, j, k, grid, x, y) = exp( - (x[i]^2 + y[j]^2) / ℓ^2)
∂xA(i, j, k, grid, x, y) = - (2 / ℓ^2) * x[i] * exp( - (x[i]^2 + y[j]^2) / ℓ^2)
∂yA(i, j, k, grid, x, y) = - (2 / ℓ^2) * y[j] * exp( - (x[i]^2 + y[j]^2) / ℓ^2)   # you had y[i], wrong index

jacobian_x = zeros(Float64, length(xᶜ), length(yᶜ))
jacobian_y = zeros(Float64, length(xᶜ), length(yᶜ))

Bx = zeros(Float64, length(xᶜ), length(yᶜ))
By = zeros(Float64, length(xᶜ), length(yᶜ))

A_num = zeros(Float64, length(xᶜ), length(yᶜ))

for i = 1:Nx, j = 1:Ny
    A_num[i, j] = A(i, j, 1, grid, xᶜ, yᶜ)
    Bx[i, j] = - ∂yᶜᶠᶜ(i, j, 1, grid, A, xᶜ, yᶜ)
    By[i, j] =   ∂xᶠᶜᶜ(i, j, 1, grid, A, xᶜ, yᶜ)

    lorentz_x[i, j] = ℑxyᶜᶠᵃ(i, j, 1, grid, ∂xᶠᶜᶜ, A, xᶜ, yᶜ) * ∂yᶜᶠᶜ(i, j, 1, grid, ∂yA, xᶜ, yᶜ)
                    - ℑxyᶜᶠᵃ(i, j, 1, grid, ∂xᶠᶜᶜ, ∂yA, xᶜ, yᶜ) * ∂yᶜᶠᶜ(i, j, 1, grid, A, xᶜ, yᶜ)
    lorentz_y[i, j] = - ∂xᶠᶜᶜ(i, j, 1, grid, A, xᶜ, yᶜ) * ℑxyᶠᶜᵃ(i, j, 1, grid, ∂yᶜᶠᶜ, ∂xA, xᶜ, yᶜ)
                      + ∂xᶠᶜᶜ(i, j, 1, grid, ∂xA, xᶜ, yᶜ) * ℑxyᶠᶜᵃ(i, j, 1, grid, ∂yᶜᶠᶜ, A, xᶜ, yᶜ)
end

plt = plot_surface(xᶜ, yᶜ, A_num, cmap=ColorMap("coolwarm"))

step = 10

#plt_A = colorbar(contourf(xᶜ, yᶜ, A_num))
#plt_lorentz = quiver(xᶜ[1:step:end], yᶜ[1:step:end], lorentz_x[1:step:end, 1:step:end], lorentz_y[1:step:end, 1:step:end], scale = 4)
#title("Contour plot of A and quiver plot of Lorentz force")
#xlim((-4, 4))
#ylim((-4, 4))

#=plt_A = colorbar(contourf(xᶜ, yᶜ, A_num))
plt_B = quiver(xᶜ[1:step:end], yᶜ[1:step:end], Bx[1:step:end, 1:step:end]', By[1:step:end, 1:step:end]', scale = 15)
title("Contour plot of A and quiver plot of magnetic field")
xlim((-4, 4))
ylim((-4, 4))=#
