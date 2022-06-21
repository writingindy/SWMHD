@inline upwind_biased_product(ũ, ψᴸ, ψᴿ) = ((ũ + abs(ũ)) * ψᴸ + (ũ - abs(ũ)) * ψᴿ) / 2

@inline ℑ³xᶠᵃᵃ(i, j, k, grid, u, args...) = @inbounds u(i, j, k, grid, args...) - δxᶠᵃᵃ(i, j, k, grid, δxᶜᵃᵃ, u, args...) / 6
@inline ℑ³xᶜᵃᵃ(i, j, k, grid, c, args...) = @inbounds c(i, j, k, grid, args...) - δxᶜᵃᵃ(i, j, k, grid, δxᶠᵃᵃ, c, args...) / 6

@inline ℑ³yᵃᶠᵃ(i, j, k, grid, v, args...) = @inbounds v(i, j, k, grid, args...) - δyᵃᶠᵃ(i, j, k, grid, δyᵃᶜᵃ, v, args...) / 6
@inline ℑ³yᵃᶜᵃ(i, j, k, grid, c, args...) = @inbounds c(i, j, k, grid, args...) - δyᵃᶜᵃ(i, j, k, grid, δyᵃᶠᵃ, c, args...) / 6

@inline symmetric_interpolate_xᶜᵃᵃ(i, j, k, grid, u, args...) = ℑxᶜᵃᵃ(i, j, k, grid, ℑ³xᶠᵃᵃ, u, args...)
@inline symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, c, args...) = ℑxᶠᵃᵃ(i, j, k, grid, ℑ³xᶜᵃᵃ, c, args...)

@inline symmetric_interpolate_yᵃᶜᵃ(i, j, k, grid, v, args...) = ℑyᵃᶜᵃ(i, j, k, grid, ℑ³yᵃᶠᵃ, v, args...)
@inline symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, c, args...) = ℑyᵃᶠᵃ(i, j, k, grid, ℑ³yᵃᶜᵃ, c, args...)

@inline left_biased_interpolate_xᶠᵃᵃ(i, j, k, grid, f, args...) = @inbounds (- 3 * f(i+1, j, k, grid, args...) + 27 * f(i, j, k, grid, args...) + 47 * f(i-1, j, k, grid, args...) - 13 * f(i-2, j, k, grid, args...) + 2 * f(i-3, j, k, grid, args...)) / 60
@inline left_biased_interpolate_xᶜᵃᵃ(i, j, k, grid, u, args...) = left_biased_interpolate_xᶠᵃᵃ(i+1, j, k, grid, u, args...)

@inline left_biased_interpolate_yᵃᶠᵃ(i, j, k, grid, f, args...) = @inbounds (- 3 * f(i, j+1, k, grid, args...) + 27 * f(i, j, k, grid, args...) + 47 * f(i, j-1, k, grid, args...) - 13 * f(i, j-2, k, grid, args...) + 2 * f(i, j-3, k, grid, args...)) / 60
@inline left_biased_interpolate_yᵃᶜᵃ(i, j, k, grid, v, args...) = left_biased_interpolate_yᵃᶠᵃ(i, j+1, k, grid, v, args...)

@inline right_biased_interpolate_xᶠᵃᵃ(i, j, k, grid, f, args...) = @inbounds (2 * f(i+2, j, k, grid, args...) - 13 * f(i+1, j, k, grid, args...) + 47 * f(i, j, k, grid, args...) + 27 * f(i-1, j, k, grid, args...) - 3 * f(i-2, j, k, grid, args...) ) / 60
@inline right_biased_interpolate_xᶜᵃᵃ(i, j, k, grid, u, args...) = right_biased_interpolate_xᶠᵃᵃ(i+1, j, k, grid, u, args...)

@inline right_biased_interpolate_yᵃᶠᵃ(i, j, k, grid, f, args...) = @inbounds (2 * f(i, j+2, k, grid, args...) - 13 * f(i, j+1, k, grid, args...) + 47 * f(i, j, k, grid, args...) + 27 * f(i, j-1, k, grid, args...) - 3 * f(i, j-2, k, grid, args...) ) / 60
@inline right_biased_interpolate_yᵃᶜᵃ(i, j, k, grid, v, args...) = right_biased_interpolate_yᵃᶠᵃ(i, j+1, k, grid, v, args...)

@inline function advective_momentum_flux_Uu(i, j, k, grid, U, u, fields)

    ũ  =    symmetric_interpolate_xᶜᵃᵃ(i, j, k, grid, U, fields.A, fields.h)
    uᴸ =  left_biased_interpolate_xᶜᵃᵃ(i, j, k, grid, u, fields.A, fields.h)
    uᴿ = right_biased_interpolate_xᶜᵃᵃ(i, j, k, grid, u, fields.A, fields.h)

    return Axᶜᶜᶜ(i, j, k, grid) * upwind_biased_product(ũ, uᴸ, uᴿ)
end

@inline function advective_momentum_flux_Vu(i, j, k, grid, V, u, fields)

    ṽ  =    symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, V, fields.A, fields.h)
    uᴸ =  left_biased_interpolate_yᵃᶠᵃ(i, j, k, grid, u, fields.A, fields.h)
    uᴿ = right_biased_interpolate_yᵃᶠᵃ(i, j, k, grid, u, fields.A, fields.h)

    return Ayᶠᶠᶜ(i, j, k, grid) * upwind_biased_product(ṽ, uᴸ, uᴿ)
end

@inline function advective_momentum_flux_Uv(i, j, k, grid, U, v, fields)

    ũ  =    symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, U, fields.A, fields.h)
    vᴸ =  left_biased_interpolate_xᶠᵃᵃ(i, j, k, grid, v, fields.A, fields.h)
    vᴿ = right_biased_interpolate_xᶠᵃᵃ(i, j, k, grid, v, fields.A, fields.h)
 
    return Axᶠᶠᶜ(i, j, k, grid) * upwind_biased_product(ũ, vᴸ, vᴿ)
end

@inline function advective_momentum_flux_Vv(i, j, k, grid, V, v, fields)

    ṽ  =    symmetric_interpolate_yᵃᶜᵃ(i, j, k, grid, V, fields.A, fields.h)
    vᴸ =  left_biased_interpolate_yᵃᶜᵃ(i, j, k, grid, v, fields.A, fields.h)
    vᴿ = right_biased_interpolate_yᵃᶜᵃ(i, j, k, grid, v, fields.A, fields.h)

    return Ayᶜᶜᶜ(i, j, k, grid) * upwind_biased_product(ṽ, vᴸ, vᴿ)
end

@inline momentum_flux_hbx_bx(i, j, k, grid, fields) = 
    @inbounds advective_momentum_flux_Uu(i, j, k, grid, hB_x, B_x, fields) / fields.h[i, j, k]

@inline momentum_flux_hby_bx(i, j, k, grid, fields) =
    @inbounds advective_momentum_flux_Vu(i, j, k, grid, hB_y, B_x, fields) / ℑxyᶠᶠᵃ(i, j, k, grid, fields.h)

@inline momentum_flux_hbx_by(i, j, k, grid, fields) =
    @inbounds advective_momentum_flux_Uv(i, j, k, grid, hB_x, B_y, fields) / ℑxyᶠᶠᵃ(i, j, k, grid, fields.h)

@inline momentum_flux_hby_by(i, j, k, grid, fields) =
    @inbounds advective_momentum_flux_Vv(i, j, k, grid, hB_y, B_y, fields) / fields.h[i, j, k]

