# Blended first and third order advection Scheme

@inline upwind_biased_product(ũ, ψᴸ, ψᴿ) = ((ũ + abs(ũ)) * ψᴸ + (ũ - abs(ũ)) * ψᴿ) / 2

@inline symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, c, args...) = ℑxᶠᵃᵃ(i, j, k, grid, c, args...)
@inline symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, c, args...) = ℑyᵃᶠᵃ(i, j, k, grid, c, args...)

@inline symmetric_interpolate_xᶜᵃᵃ(i, j, k, grid, u, args...) = ℑxᶜᵃᵃ(i, j, k, grid, u, args...)
@inline symmetric_interpolate_yᵃᶜᵃ(i, j, k, grid, v, args...) = ℑyᵃᶜᵃ(i, j, k, grid, v, args...)

# First order interpolation functions
@inline _left_biased_interpolate_xᶠᵃᵃ(i, j, k, grid, c, args...) = @inbounds c(i-1, j, k, grid, args...) 
@inline _left_biased_interpolate_yᵃᶠᵃ(i, j, k, grid, c, args...) = @inbounds c(i, j-1, k, grid, args...)

@inline _left_biased_interpolate_xᶜᵃᵃ(i, j, k, grid, u, args...) = _left_biased_interpolate_xᶠᵃᵃ(i+1, j, k, grid, u, args...)
@inline _left_biased_interpolate_yᵃᶜᵃ(i, j, k, grid, v, args...) = _left_biased_interpolate_yᵃᶠᵃ(i, j+1, k, grid, v, args...)

@inline _right_biased_interpolate_xᶠᵃᵃ(i, j, k, grid, c, args...) = @inbounds c(i, j, k, grid, args...) 
@inline _right_biased_interpolate_yᵃᶠᵃ(i, j, k, grid, c, args...) = @inbounds c(i, j, k, grid, args...)

@inline _right_biased_interpolate_xᶜᵃᵃ(i, j, k, grid, u, args...) = _right_biased_interpolate_xᶠᵃᵃ(i+1, j, k, grid, u, args...)
@inline _right_biased_interpolate_yᵃᶜᵃ(i, j, k, grid, v, args...) = _right_biased_interpolate_yᵃᶠᵃ(i, j+1, k, grid, v, args...)

# Third order interpolation functions
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
    
    if topology(grid, 1) == Bounded && i == (grid.Nx + 1)
        uᴸ = _left_biased_interpolate_xᶜᵃᵃ(i, j, k, grid, u, args...)
        uᴿ = _right_biased_interpolate_xᶜᵃᵃ(i, j, k, grid, u, args...) 
    elseif topology(grid, 1) == Bounded && i == grid.Nx
        uᴸ = left_biased_interpolate_xᶜᵃᵃ(i, j, k, grid, u, args...)
        uᴿ = _right_biased_interpolate_xᶜᵃᵃ(i, j, k, grid, u, args...)
    else
        uᴸ = left_biased_interpolate_xᶜᵃᵃ(i, j, k, grid, u, args...)
        uᴿ = right_biased_interpolate_xᶜᵃᵃ(i, j, k, grid, u, args...) 
    end

    return Axᶜᶜᶜ(i, j, k, grid) * upwind_biased_product(ũ, uᴸ, uᴿ)
end

@inline function advective_lorentz_flux_hBy_bx(i, j, k, grid, V, u, args...)

    ṽ  =    symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, V, args...)

    if topology(grid, 2) == Bounded && j == 1  
        uᴸ = _left_biased_interpolate_yᵃᶠᵃ(i, j, k, grid, u, args...)
        uᴿ = right_biased_interpolate_yᵃᶠᵃ(i, j, k, grid, u, args...)
    elseif topology(grid, 2) == Bounded && j == (grid.Ny + 1)
        uᴸ =  left_biased_interpolate_yᵃᶠᵃ(i, j, k, grid, u, args...)
        uᴿ = _right_biased_interpolate_yᵃᶠᵃ(i, j, k, grid, u, args...)
    else
        uᴸ =  left_biased_interpolate_yᵃᶠᵃ(i, j, k, grid, u, args...)
        uᴿ = right_biased_interpolate_yᵃᶠᵃ(i, j, k, grid, u, args...)
    end

    return Ayᶠᶠᶜ(i, j, k, grid) * upwind_biased_product(ṽ, uᴸ, uᴿ)
end

@inline function advective_lorentz_flux_hBx_by(i, j, k, grid, U, v, args...)

    ũ  =    symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, U, args...)

    if topology(grid, 1) == Bounded && i == 1
        vᴸ = _left_biased_interpolate_xᶠᵃᵃ(i, j, k, grid, v, args...)
        vᴿ = right_biased_interpolate_xᶠᵃᵃ(i, j, k, grid, v, args...)
    elseif topology(grid, 1) == Bounded && i == (grid.Nx + 1)
        vᴸ = left_biased_interpolate_xᶠᵃᵃ(i, j, k, grid, v, args...)
        vᴿ = _right_biased_interpolate_xᶠᵃᵃ(i, j, k, grid, v, args...)
    else
        vᴸ =  left_biased_interpolate_xᶠᵃᵃ(i, j, k, grid, v, args...)
        vᴿ = right_biased_interpolate_xᶠᵃᵃ(i, j, k, grid, v, args...)
    end
    
    return Axᶠᶠᶜ(i, j, k, grid) * upwind_biased_product(ũ, vᴸ, vᴿ)
end

@inline function advective_lorentz_flux_hBy_by(i, j, k, grid, V, v, args...)

    ṽ  =    symmetric_interpolate_yᵃᶜᵃ(i, j, k, grid, V, args...)
    
    if topology(grid, 2) == Bounded && j == (grid.Ny + 1)
        vᴸ = _left_biased_interpolate_yᵃᶜᵃ(i, j, k, grid, v, args...)
        vᴿ = _right_biased_interpolate_yᵃᶜᵃ(i, j, k, grid, v, args...)
    elseif topology(grid, 2) == Bounded && j == grid.Ny
        vᴸ = left_biased_interpolate_yᵃᶜᵃ(i, j, k, grid, v, args...)
        vᴿ = _right_biased_interpolate_yᵃᶜᵃ(i, j, k, grid, v, args...)
    else
        vᴸ = left_biased_interpolate_yᵃᶜᵃ(i, j, k, grid, v, args...)
        vᴿ = right_biased_interpolate_yᵃᶜᵃ(i, j, k, grid, v, args...)
    end

    return Ayᶜᶜᶜ(i, j, k, grid) * upwind_biased_product(ṽ, vᴸ, vᴿ)
end

function Bx(i, j, k, grid, A, h)
    return -ℑxyᶠᶜᵃ(i, j, k, grid, ∂yᶜᶠᶜ, A) / ℑxᶠᵃᵃ(i, j, k, grid, h)
end

function By(i, j, k, grid, A, h)
    return ℑxyᶜᶠᵃ(i, j, k, grid, ∂xᶠᶜᶜ, A) / ℑyᵃᶠᵃ(i, j, k, grid, h)
end

function hBx(i, j, k, grid, A, h)
    return -ℑxyᶠᶜᵃ(i, j, k, grid, ∂yᶜᶠᶜ, A)
end

function hBy(i, j, k, grid, A, h)
    return ℑxyᶜᶠᵃ(i, j, k, grid, ∂xᶠᶜᶜ, A)
end

@inline lorentz_flux_hBx_bx(i, j, k, grid, fields) = 
    @inbounds advective_lorentz_flux_hBx_bx(i, j, k, grid, hBx, Bx, fields.A, fields.h)

@inline lorentz_flux_hBy_bx(i, j, k, grid, fields) =
    @inbounds advective_lorentz_flux_hBy_bx(i, j, k, grid, hBy, Bx, fields.A, fields.h)

@inline lorentz_flux_hBx_by(i, j, k, grid, fields) =
    @inbounds advective_lorentz_flux_hBx_by(i, j, k, grid, hBx, By, fields.A, fields.h)

@inline lorentz_flux_hBy_by(i, j, k, grid, fields) =
    @inbounds advective_lorentz_flux_hBy_by(i, j, k, grid, hBy, By, fields.A, fields.h)

function div_lorentz_x(i, j, k, grid, clock, fields)
    return ((1/Azᶠᶜᶜ(i, j, k, grid)) * (δxᶠᵃᵃ(i, j, k, grid, lorentz_flux_hBx_bx, fields) 
                                     + δyᵃᶜᵃ(i, j, k, grid, lorentz_flux_hBy_bx, fields)))
end

function div_lorentz_y(i, j, k, grid, clock, fields)
    return ((1/Azᶜᶠᶜ(i, j, k, grid)) * (δxᶜᵃᵃ(i, j, k, grid, lorentz_flux_hBx_by, fields) 
                                     + δyᵃᶠᵃ(i, j, k, grid, lorentz_flux_hBy_by, fields)))
end
