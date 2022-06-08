using Oceananigans.Operators
using PyPlot
using LinearAlgebra
using Polynomials
using Printf

  A(i, j, k, grid, x, y) =   A₀                    * exp( - (x[i]^2 + y[j]^2) / ℓ^2)
∂xA(i, j, k, grid, x, y) = - A₀ * (2 / ℓ^2) * x[i] * exp( - (x[i]^2 + y[j]^2) / ℓ^2)
∂yA(i, j, k, grid, x, y) = - A₀ * (2 / ℓ^2) * y[j] * exp( - (x[i]^2 + y[j]^2) / ℓ^2)

∂xxA(i, j, k, grid, x, y) =  A₀ * ((4 * x[i]^2 - 2*ℓ^2) / ℓ^4) * exp( - (x[i]^2 + y[j]^2) / ℓ^2)
∂xyA(i, j, k, grid, x, y) =  A₀ * ((4 * x[i] * y[j]) / ℓ^4)    * exp( - (x[i]^2 + y[j]^2) / ℓ^2)
∂yyA(i, j, k, grid, x, y) =  A₀ * ((4 * y[j]^2 - 2*ℓ^2) / ℓ^4) * exp( - (x[i]^2 + y[j]^2) / ℓ^2)

∂xA_num(i, j, k, grid, A, x, y) = ℑxᶜᵃᵃ(i, j, k, grid, ∂xᶠᶜᶜ, A, x, y)
∂yA_num(i, j, k, grid, A, x, y) = ℑyᵃᶜᵃ(i, j, k, grid, ∂yᶜᶠᶜ, A, x, y)

 ℓ = 2
A₀ = -1

Lx, Ly = 10,  10

Ns = [50 100 200 400]

error_Bx        = Dict()
error_By        = Dict()
error_lorentz_x = Dict()
error_lorentz_y = Dict()

step = 10

for N in Ns
  @printf("Computing the jacobian for a resolution of N = %d \n", N)
  
  local Nx, Ny = N, N
  
  local grid = RectilinearGrid(; size = (Nx, Ny),  x = (-Lx/2, Lx/2), y = (-Ly/2, Ly/2),
                      topology = (Periodic, Periodic, Flat))
  
  local xᶜ, xᶠ = grid.xᶜᵃᵃ, grid.xᶠᵃᵃ
  local yᶜ, yᶠ = grid.yᵃᶜᵃ, grid.yᵃᶠᵃ
  
  local Bx     = zeros(Float64, length(xᶜ), length(yᶜ))
  local By     = zeros(Float64, length(xᶜ), length(yᶜ))
  local A_num  = zeros(Float64, length(xᶜ), length(yᶜ))
  local Bx_num = zeros(Float64, length(xᶜ), length(yᶜ))
  local By_num = zeros(Float64, length(xᶜ), length(yᶜ))

  local lorentz_x     = zeros(Float64, length(xᶜ), length(yᶜ))
  local lorentz_y     = zeros(Float64, length(xᶜ), length(yᶜ))
  local lorentz_x_num = zeros(Float64, length(xᶜ), length(yᶜ))
  local lorentz_y_num = zeros(Float64, length(xᶜ), length(yᶜ))
                      
  for i = 1:Nx, j = 1:Ny
    Bx[i, j] = - ∂yA(i, j, 1, grid, xᶜ, yᶠ)
    By[i, j] =   ∂xA(i, j, 1, grid, xᶠ, yᶜ)
    
    lorentz_x[i, j] = ((∂xA(i, j, 1, grid, xᶠ, yᶜ) * ∂yyA(i, j, 1, grid, xᶠ, yᶜ)) 
                     - (∂yA(i, j, 1, grid, xᶠ, yᶜ) * ∂xyA(i, j, 1, grid, xᶠ, yᶜ)))
    lorentz_y[i, j] = ((∂yA(i, j, 1, grid, xᶜ, yᶠ) * ∂xxA(i, j, 1, grid, xᶜ, yᶠ)) 
                     - (∂xA(i, j, 1, grid, xᶜ, yᶠ) * ∂xyA(i, j, 1, grid, xᶜ, yᶠ)))
    
    A_num[i, j]  =      A(i, j, 1, grid, xᶜ, yᶜ)
    Bx_num[i, j] = -∂yᶜᶠᶜ(i, j, 1, grid, A, xᶜ, yᶜ)
    By_num[i, j] =  ∂xᶠᶜᶜ(i, j, 1, grid, A, xᶜ, yᶜ)

    lorentz_x_num[i, j] = ((∂xᶠᶜᶜ(i, j, 1, grid, A, xᶜ, yᶜ) * ℑxyᶠᶜᵃ(i, j, 1, grid, ∂yᶜᶠᶜ, ∂yA_num, A, xᶜ, yᶜ)) 
                        - (ℑxyᶠᶜᵃ(i, j, 1, grid, ∂yᶜᶠᶜ, A, xᶜ, yᶜ) * ∂xᶠᶜᶜ(i, j, 1, grid, ∂yA_num, A, xᶜ, yᶜ)))
    lorentz_y_num[i, j] = ((∂yᶜᶠᶜ(i, j, 1, grid, A, xᶜ, yᶜ) * ℑxyᶜᶠᵃ(i, j, 1, grid, ∂xᶠᶜᶜ, ∂xA_num, A, xᶜ, yᶜ)) 
                         - (∂yᶜᶠᶜ(i, j, 1, grid, ∂xA_num, A, xᶜ, yᶜ) * ℑxyᶜᶠᵃ(i, j, 1, grid, ∂xᶠᶜᶜ, A, xᶜ, yᶜ)))

  end

  error_Bx[N]        = maximum(abs, Bx .- Bx_num)
  error_By[N]        = maximum(abs, By .- By_num)
  error_lorentz_x[N] = maximum(abs, lorentz_x .- lorentz_x_num)
  error_lorentz_y[N] = maximum(abs, lorentz_y .- lorentz_y_num)
  
  if N == Ns[end]

    figure()
    local plt_A1 = colorbar(contourf(xᶜ, yᶜ, A_num))
    local plt_lorentz = quiver(xᶜ[1:step:end], yᶜ[1:step:end], lorentz_x_num[1:step:end, 1:step:end]', lorentz_y_num[1:step:end, 1:step:end]', scale = 3)
    title("Potential Function and Lorentz Force")
    xlim((-4, 4))
    ylim((-4, 4))
    @info "Plotting the Lorentz force  in figure Lorentz_force.png"
    savefig("Lorentz_Force.png")

    figure()
    local plt_A2 = colorbar(contourf(xᶜ, yᶜ, A_num))
    local plt_B = quiver(xᶜ[1:step:end], yᶜ[1:step:end], Bx[1:step:end, 1:step:end]', By[1:step:end, 1:step:end]', scale = 15)
    title("Potential Function and Magnetic Field")
    xlim((-4, 4))
    ylim((-4, 4))
    @info "Plotting the Magnetic Field in figure Lorentz_force.png"
    savefig("Magnetic_Field.png")
  end
end

best_fit_Bx        = fit(log10.(Ns[1:end]), log10.([error_Bx[N]        for N in Ns][1:end]), 1)
best_fit_By        = fit(log10.(Ns[1:end]), log10.([error_By[N]        for N in Ns][1:end]), 1)
best_fit_lorentz_x = fit(log10.(Ns[1:end]), log10.([error_lorentz_x[N] for N in Ns][1:end]), 1)
best_fit_lorentz_y = fit(log10.(Ns[1:end]), log10.([error_lorentz_y[N] for N in Ns][1:end]), 1)

@printf("\n")
@printf("Error for Bx        is of order = %3.2f \n", -best_fit_Bx[1])
@printf("Error for By        is of order = %3.2f \n", -best_fit_By[1])
@printf("Error for lorentz_y is of order = %3.2f \n", -best_fit_lorentz_x[1])
@printf("Error for lorentz_y is of order = %3.2f \n", -best_fit_lorentz_y[1])
@printf("\n")
