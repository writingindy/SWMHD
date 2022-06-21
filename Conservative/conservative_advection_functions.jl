function B_x(i, j, k, grid, A, h)
    return -ℑxyᶠᶜᵃ(i, j, k, grid, ∂yᶜᶠᶜ, A) / ℑxᶠᵃᵃ(i, j, k, grid, h)
end

function B_y(i, j, k, grid, A, h)
    return ℑxyᶜᶠᵃ(i, j, k, grid, ∂xᶠᶜᶜ, A) / ℑyᵃᶠᵃ(i, j, k, grid, h)
end

function hB_x(i, j, k, grid, A, h)
    return -ℑxyᶠᶜᵃ(i, j, k, grid, ∂yᶜᶠᶜ, A)
end

function hB_y(i, j, k, grid, A, h)
    return ℑxyᶜᶠᵃ(i, j, k, grid, ∂xᶠᶜᶜ, A)
end

@inline advective_momentum_flux_Uu(i, j, k, grid, U, u, args...) = ℑxᶜᵃᵃ(i, j, k, grid, Ax_qᶠᶜᶜ, U, args...) * ℑxᶜᵃᵃ(i, j, k, grid, u, args...)
@inline advective_momentum_flux_Vu(i, j, k, grid, V, u, args...) = ℑxᶠᵃᵃ(i, j, k, grid, Ay_qᶜᶠᶜ, V, args...) * ℑyᵃᶠᵃ(i, j, k, grid, u, args...)

@inline advective_momentum_flux_Uv(i, j, k, grid, U, v, args...) = ℑyᵃᶠᵃ(i, j, k, grid, Ax_qᶠᶜᶜ, U, args...) * ℑxᶠᵃᵃ(i, j, k, grid, v, args...)
@inline advective_momentum_flux_Vv(i, j, k, grid, V, v, args...) = ℑyᵃᶜᵃ(i, j, k, grid, Ay_qᶜᶠᶜ, V, args...) * ℑyᵃᶜᵃ(i, j, k, grid, v, args...)


@inline momentum_flux_hbx_bx(i, j, k, grid, fields) = 
    @inbounds advective_momentum_flux_Uu(i, j, k, grid, hB_x, B_x, fields.A, fields.h)

@inline momentum_flux_hby_bx(i, j, k, grid, fields) =
    @inbounds advective_momentum_flux_Vu(i, j, k, grid, hB_y, B_x, fields.A, fields.h)

@inline momentum_flux_hbx_by(i, j, k, grid, fields) =
    @inbounds advective_momentum_flux_Uv(i, j, k, grid, hB_x, B_y, fields.A, fields.h)

@inline momentum_flux_hby_by(i, j, k, grid, fields) =
    @inbounds advective_momentum_flux_Vv(i, j, k, grid, hB_y, B_y, fields.A, fields.h)


function div_lorentz_x(i, j, k, grid, clock, fields)
    return 1 / Vᶠᶜᶜ(i, j, k, grid) * (δxᶠᵃᵃ(i, j, k, grid, momentum_flux_hbx_bx, fields) + δyᵃᶜᵃ(i, j, k, grid, momentum_flux_hby_bx, fields))
end

function div_lorentz_y(i, j, k, grid, clock, fields)
    return 1 / Vᶜᶠᶜ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, momentum_flux_hbx_by, fields) + δyᵃᶠᵃ(i, j, k, grid, momentum_flux_hby_by, fields))
end


