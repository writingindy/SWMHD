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

@inline momentum_flux_hbx_bx(i, j, k, grid, advection, fields) = 
    @inbounds _advective_momentum_flux_Uu(i, j, k, grid, advection, hB_x, B_x, fields.A, fields.h)

@inline momentum_flux_hby_bx(i, j, k, grid, advection, fields) =
    @inbounds _advective_momentum_flux_Vu(i, j, k, grid, advection, hB_y, B_x, fields.A, fields.h)

@inline momentum_flux_hbx_by(i, j, k, grid, advection, fields) =
    @inbounds _advective_momentum_flux_Uv(i, j, k, grid, advection, hB_x, B_y, fields.A, fields.h)

@inline momentum_flux_hby_by(i, j, k, grid, advection, fields) =
    @inbounds _advective_momentum_flux_Vv(i, j, k, grid, advection, hB_y, B_y, fields.A, fields.h)



function div_lorentz_x(i, j, k, grid, clock, advection, fields)
    return 1 / Vᶠᶜᶜ(i, j, k, grid) * (δxᶠᵃᵃ(i, j, k, grid, momentum_flux_hbx_bx, advection, fields) + δyᵃᶜᵃ(i, j, k, grid, momentum_flux_hby_bx, advection, fields))
end

function div_lorentz_y(i, j, k, grid, clock, advection, fields)
    return 1 / Vᶜᶠᶜ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, momentum_flux_hbx_by, advection, fields) + δyᵃᶠᵃ(i, j, k, grid, momentum_flux_hby_by, advection, fields))
end

