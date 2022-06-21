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

function div_lorentz_x(i, j, k, grid, clock, fields)
    return 1 / Vᶠᶜᶜ(i, j, k, grid) * (δxᶠᵃᵃ(i, j, k, grid, momentum_flux_hbx_bx, fields) + δyᵃᶜᵃ(i, j, k, grid, momentum_flux_hby_bx, fields))
end

function div_lorentz_y(i, j, k, grid, clock, fields)
    return 1 / Vᶜᶠᶜ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, momentum_flux_hbx_by, fields) + δyᵃᶠᵃ(i, j, k, grid, momentum_flux_hby_by, fields))
end


