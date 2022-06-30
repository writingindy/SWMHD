using Oceananigans
using Oceananigans.Models.ShallowWaterModels: ConservativeFormulation
using Oceananigans.Advection: VelocityStencil, VorticityStencil
using Oceananigans.Operators
using Statistics, JLD2, Printf
using Polynomials
using PyPlot

using Oceananigans.Grids: AbstractGrid
using Oceananigans.Operators: Ax_qᶠᶜᶜ, Ay_qᶜᶠᶜ

A(i, j, k, grid, x, y) = exp(-(x[i]^2 + y[j]^2))

exact_lorentz_x_func(i, j, k, grid, x, y) = (-4*x[i])*exp(-2*(x[i]^2 + y[j]^2))
exact_lorentz_y_func(i, j, k, grid, x, y) = (-4*y[j])*exp(-2*(x[i]^2 + y[j]^2))

Lx, Ly = 10, 10
Ns = [50 100 200]

error_lorentz_x = Dict()
error_lorentz_y = Dict()

step = 10

# Upwind Biased Third Order Advection Scheme

@inline upwind_biased_product(ũ, ψᴸ, ψᴿ) = ((ũ + abs(ũ)) * ψᴸ + (ũ - abs(ũ)) * ψᴿ) / 2

@inline symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, c, args...) = ℑxᶠᵃᵃ(i, j, k, grid, c, args...)
@inline symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, c, args...) = ℑyᵃᶠᵃ(i, j, k, grid, c, args...)

@inline symmetric_interpolate_xᶜᵃᵃ(i, j, k, grid, u, args...) = ℑxᶜᵃᵃ(i, j, k, grid, u, args...)
@inline symmetric_interpolate_yᵃᶜᵃ(i, j, k, grid, v, args...) = ℑyᵃᶜᵃ(i, j, k, grid, v, args...)

@inline left_biased_interpolate_xᶠᵃᵃ(i, j, k, grid, c, args...) = @inbounds (2 * c(i, j, k, grid, args...) + 5 * c(i-1, j, k, grid, args...) - c(i-2, j, k, grid, args...)) / 6
@inline left_biased_interpolate_yᵃᶠᵃ(i, j, k, grid, c, args...) = @inbounds (2 * c(i, j, k, grid, args...) + 5 * c(i, j-1, k, grid, args...) - c(i, j-2, k, grid, args...)) / 6

@inline left_biased_interpolate_xᶜᵃᵃ(i, j, k, grid, u, args...) = left_biased_interpolate_xᶠᵃᵃ(i+1, j, k, grid, u, args...)
@inline left_biased_interpolate_yᵃᶜᵃ(i, j, k, grid, v, args...) = left_biased_interpolate_yᵃᶠᵃ(i, j+1, k, grid, v, args...)

@inline right_biased_interpolate_xᶠᵃᵃ(i, j, k, grid, c, args...) = @inbounds (- c(i+1, j, k, grid, args...) + 5 * c(i, j, k, grid, args...) + 2 * c(i-1, j, k, grid, args...)) / 6
@inline right_biased_interpolate_yᵃᶠᵃ(i, j, k, grid, c, args...) = @inbounds (- c(i, j+1, k, grid, args...) + 5 * c(i, j, k, grid, args...) + 2 * c(i, j-1, k, grid, args...)) / 6

@inline right_biased_interpolate_xᶜᵃᵃ(i, j, k, grid, u, args...) = right_biased_interpolate_xᶠᵃᵃ(i+1, j, k, grid, u, args...)
@inline right_biased_interpolate_yᵃᶜᵃ(i, j, k, grid, v, args...) = right_biased_interpolate_yᵃᶠᵃ(i, j+1, k, grid, v, args...)

@inline function advective_lorentz_flux_hBx_bx(i, j, k, grid, U, u, args...)

    ũ  =    symmetric_interpolate_xᶜᵃᵃ(i, j, k, grid, U, args...)
    uᴸ =  left_biased_interpolate_xᶜᵃᵃ(i, j, k, grid, u, args...)
    uᴿ = right_biased_interpolate_xᶜᵃᵃ(i, j, k, grid, u, args...)

    return Axᶜᶜᶜ(i, j, k, grid) * upwind_biased_product(ũ, uᴸ, uᴿ)
end

@inline function advective_lorentz_flux_hBy_bx(i, j, k, grid, V, u, args...)

    ṽ  =    symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, V, args...)
    uᴸ =  left_biased_interpolate_yᵃᶠᵃ(i, j, k, grid, u, args...)
    uᴿ = right_biased_interpolate_yᵃᶠᵃ(i, j, k, grid, u, args...)

    return Ayᶠᶠᶜ(i, j, k, grid) * upwind_biased_product(ṽ, uᴸ, uᴿ)
end

@inline function advective_lorentz_flux_hBx_by(i, j, k, grid, U, v, args...)

    ũ  =    symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, U, args...)
    vᴸ =  left_biased_interpolate_xᶠᵃᵃ(i, j, k, grid, v, args...)
    vᴿ = right_biased_interpolate_xᶠᵃᵃ(i, j, k, grid, v, args...)
 
    return Axᶠᶠᶜ(i, j, k, grid) * upwind_biased_product(ũ, vᴸ, vᴿ)
end

@inline function advective_lorentz_flux_hBy_by(i, j, k, grid, V, v, args...)

    ṽ  =    symmetric_interpolate_yᵃᶜᵃ(i, j, k, grid, V, args...)
    vᴸ =  left_biased_interpolate_yᵃᶜᵃ(i, j, k, grid, v, args...)
    vᴿ = right_biased_interpolate_yᵃᶜᵃ(i, j, k, grid, v, args...)

    return Ayᶜᶜᶜ(i, j, k, grid) * upwind_biased_product(ṽ, vᴸ, vᴿ)
end

function Bx(i, j, k, grid, A, args...)
    return -ℑxyᶠᶜᵃ(i, j, k, grid, ∂yᶜᶠᶜ, A, args...)
end

function By(i, j, k, grid, A, args...)
    return ℑxyᶜᶠᵃ(i, j, k, grid, ∂xᶠᶜᶜ, A, args...)
end

function hBx(i, j, k, grid, A, args...)
    return -ℑxyᶠᶜᵃ(i, j, k, grid, ∂yᶜᶠᶜ, A, args...)
end

function hBy(i, j, k, grid, A, args...)
    return ℑxyᶜᶠᵃ(i, j, k, grid, ∂xᶠᶜᶜ, A, args...)
end


@inline lorentz_flux_hBx_bx(i, j, k, grid, A, args...) = 
    @inbounds advective_lorentz_flux_hBx_bx(i, j, k, grid, hBx, Bx, A, args...)

@inline lorentz_flux_hBy_bx(i, j, k, grid, A, args...) =
    @inbounds advective_lorentz_flux_hBy_bx(i, j, k, grid, hBy, Bx, A, args...)

@inline lorentz_flux_hBx_by(i, j, k, grid, A, args...) =
    @inbounds advective_lorentz_flux_hBx_by(i, j, k, grid, hBx, By, A, args...)

@inline lorentz_flux_hBy_by(i, j, k, grid, A, args...) =
    @inbounds advective_lorentz_flux_hBy_by(i, j, k, grid, hBy, By, A, args...)

function div_lorentz_x(i, j, k, grid, A, args...)
    return ((1/Azᶠᶜᶜ(i, j, k, grid)) * (δxᶠᵃᵃ(i, j, k, grid, lorentz_flux_hBx_bx, A, args...)
                                     + δyᵃᶜᵃ(i, j, k, grid, lorentz_flux_hBy_bx, A, args...)))
end

function div_lorentz_y(i, j, k, grid, A, args...)
    return ((1/Azᶜᶠᶜ(i, j, k, grid)) * (δxᶜᵃᵃ(i, j, k, grid, lorentz_flux_hBx_by, A, args...) 
                                     + δyᵃᶠᵃ(i, j, k, grid, lorentz_flux_hBy_by, A, args...)))
end

for N in Ns
    @printf("Computing the lorentz force for a resolution of N = %d \n", N)
  
    local Nx, Ny = N, N
    
    local grid = RectilinearGrid(; size = (Nx, Ny),  x = (-Lx/2, Lx/2), y = (-Ly/2, Ly/2), topology = (Periodic, Periodic, Flat))
    
    local xᶜ, xᶠ = grid.xᶜᵃᵃ, grid.xᶠᵃᵃ
    local yᶜ, yᶠ = grid.yᵃᶜᵃ, grid.yᵃᶠᵃ
    
    local advection_lorentz_x = zeros(Float64, length(xᶜ), length(yᶜ))
    local advection_lorentz_y = zeros(Float64, length(xᶜ), length(yᶜ))
    
    local exact_lorentz_x = zeros(Float64, length(xᶜ), length(yᶜ))
    local exact_lorentz_y = zeros(Float64, length(xᶜ), length(yᶜ))
    

    local error_x = zeros(Float64, length(xᶜ), length(yᶜ))
    local error_y = zeros(Float64, length(xᶜ), length(yᶜ))
    
    for i = 1:Nx, j = 1:Ny
        advection_lorentz_x[i, j] = div_lorentz_x(i, j, 1, grid, A, xᶜ, yᶜ)
        advection_lorentz_y[i, j] = div_lorentz_y(i, j, 1, grid, A, xᶜ, yᶜ)
    
        exact_lorentz_x[i, j] = exact_lorentz_x_func(i, j, 1, grid, xᶠ, yᶜ)
        exact_lorentz_y[i, j] = exact_lorentz_y_func(i, j, 1, grid, xᶜ, yᶠ)

        error_x[i, j] = maximum(abs(advection_lorentz_x[i, j] - exact_lorentz_x[i, j]))
        error_y[i, j] = maximum(abs(advection_lorentz_y[i, j] - exact_lorentz_y[i, j]))
    
    end

    error_lorentz_x[N] = maximum(abs, advection_lorentz_x .- exact_lorentz_x)
    error_lorentz_y[N] = maximum(abs, advection_lorentz_y .- exact_lorentz_y)

    if N == Ns[end]
        figure()
        local plt_x = colorbar(contourf(xᶜ, yᶜ, error_x))
        title("Error in lorentz_x")
        savefig("lorentz_x_error.png")

        figure()
        local plt_y = colorbar(contourf(xᶜ, yᶜ, error_y))
        title("Error in lorentz_y")
        savefig("lorentz_y_error.png")
    
    end
end

best_fit_lorentz_x = fit(log10.(Ns[1:end]), log10.([error_lorentz_x[N] for N in Ns][1:end]), 1)
best_fit_lorentz_y = fit(log10.(Ns[1:end]), log10.([error_lorentz_y[N] for N in Ns][1:end]), 1)

@printf("\n")
@printf("Error for lorentz_x is of order = %3.2f \n", -best_fit_lorentz_x[1])
@printf("Error for lorentz_y is of order = %3.2f \n", -best_fit_lorentz_y[1])
@printf("\n")