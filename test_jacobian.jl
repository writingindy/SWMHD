using Oceananigans
using Oceananigans.Operators
using PyPlot

Lx, Ly = 10,  10

Ns = [50 100 200 400]

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

∂xA_num(i, j, k, grid, A, x, y) = ℑxᶜᵃᵃ(i, j, k, grid, ∂xᶠᶜᶜ, A, x, y)
∂yA_num(i, j, k, grid, A, x, y) = ℑyᵃᶜᵃ(i, j, k, grid, ∂yᶜᶠᶜ, A, x, y)

error_jacobian_x = Dict()
error_jacobian_y = Dict()

for N in Ns
    @printf("Computing the jacobian for a resolution of N = %d \n", N)

    local Nx, Ny = N, N
  
    local grid = RectilinearGrid(; size = (Nx, Ny),  x = (-Lx/2, Lx/2), y = (-Ly/2, Ly/2), topology = (Periodic, Periodic, Flat))
  
    local xᶜ, xᶠ = grid.xᶜᵃᵃ, grid.xᶠᵃᵃ
    local yᶜ, yᶠ = grid.yᵃᶜᵃ, grid.yᵃᶠᵃ

    local  jacobian_x_exact = zeros(Float64, length(xᶠ), length(yᶠ))
    local jacobian_x_approx = zeros(Float64, length(xᶜ), length(yᶜ))

    local jacobian_y_exact = zeros(Float64, length(xᶠ), length(yᶠ))
    local jacobian_y_approx = zeros(Float64, length(xᶜ), length(yᶜ))

    for i = 1:Nx, j = 1:Ny
        jacobian_x_approx[i, j] = ((∂xᶠᶜᶜ(i, j, 1, grid, A, xᶜ, yᶜ) * ℑxyᶠᶜᵃ(i, j, 1, grid, ∂yᶜᶠᶜ, ∂xA_num, A, xᶜ, yᶜ))
                                - (∂xᶠᶜᶜ(i, j, 1, grid, ∂xA_num, A, xᶜ, yᶜ) * ℑxyᶠᶜᵃ(i, j, 1, grid, ∂yᶜᶠᶜ, A, xᶜ, yᶜ)))
        jacobian_y_approx[i, j] = ((ℑxyᶜᶠᵃ(i, j, 1, grid, ∂xᶠᶜᶜ, A, xᶜ, yᶜ) * ∂yᶜᶠᶜ(i, j, 1, grid, ∂yA_num, A, xᶜ, yᶜ))
                                - (ℑxyᶜᶠᵃ(i, j, 1, grid, ∂xᶠᶜᶜ, ∂yA_num, A, xᶜ, yᶜ) * ∂yᶜᶠᶜ(i, j, 1, grid, A, xᶜ, yᶜ)))
    
        jacobian_x_exact[i, j] = ((∂A_x(i, j, 1, grid, xᶠ, yᶜ) * ∂A_xy(i, j, 1, grid, xᶠ, yᶜ)) 
                                - (∂A_xx(i, j, 1, grid, xᶠ, yᶜ) * ∂A_y(i, j, 1, grid, xᶠ, yᶜ)))
        jacobian_y_exact[i, j] = ((∂A_x(i, j, 1, grid, xᶜ, yᶠ) * ∂A_yy(i, j, 1, grid, xᶜ, yᶠ))
                                - (∂A_yx(i, j, 1, grid, xᶜ, yᶠ) * ∂A_y(i, j, 1, grid, xᶜ, yᶠ)))
    end

    error_jacobian_x[N] = maximum(abs, jacobian_x_exact .- jacobian_x_approx)
    error_jacobian_y[N] = maximum(abs, jacobian_y_exact .- jacobian_y_approx)

end

best_fit_jacobian_x = fit(log10.(Ns[1:end]), log10.([error_jacobian_x[N] for N in Ns][1:end]), 1)
best_fit_jacobian_y = fit(log10.(Ns[1:end]), log10.([error_jacobian_y[N] for N in Ns][1:end]), 1)

@printf("\n")
@printf("Error for jacobian_x is of order = %3.2f \n", -best_fit_jacobian_x[1])
@printf("Error for jacobian_y is of order = %3.2f \n", -best_fit_jacobian_y[1])
@printf("\n")
