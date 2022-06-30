function By(i, j, k, grid, A, h)
    return ℑxᶜᵃᵃ(i, j, k, grid, ∂xᶠᶜᶜ, A, args...)/h[i, j, k]
end

function Bx(i, j, k, grid, A, h)
    return -ℑyᵃᶜᵃ(i, j, k, grid, ∂yᶜᶠᶜ, A, args...)/h[i, j, k]
end


function jacobian_x(i, j, k, grid, A, h)
    return ((∂xᶠᶜᶜ(i, j, 1, grid, A, h)*ℑxyᶠᶜᵃ(i, j, k, grid, ∂yᶜᶠᶜ, Bx, A, h)) 
            - (ℑxyᶠᶜᵃ(i, j, 1, grid, ∂yᶜᶠᶜ, A, h)*∂xᶠᶜᶜ(i, j, k, grid, Bx, A, h)))
end

function jacobian_y(i, j, k, grid, A, h)
    return ((ℑxyᶜᶠᵃ(i, j, k, grid, ∂xᶠᶜᶜ, A, h)*∂yᶜᶠᶜ(i, j, k, grid, By, A, h))
            - (∂yᶜᶠᶜ(i, j, k, grid, A, h)*ℑxyᶜᶠᵃ(i, j, k, grid, ∂xᶠᶜᶜ, By, A, h)))
end

function lorentz_force_func_x(i, j, k, grid, clock, fields)
    return (1/ℑxᶠᵃᵃ(i, j, k, grid, fields.h))*(jacobian_x(i, j, k, grid, fields.A, fields.h))
end

function lorentz_force_func_y(i, j, k, grid, clock, fields)
    return (1/ℑyᵃᶠᵃ(i, j, k, grid, fields.h))*(jacobian_y(i, j, k, grid, fields.A, fields.h))
end
