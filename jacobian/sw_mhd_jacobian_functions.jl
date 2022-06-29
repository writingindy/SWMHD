function ∂xA_over_h(i, j, k, grid, A, h)
    return ℑxᶜᵃᵃ(i, j, k, grid, ∂xᶠᶜᶜ, A)/h[i, j, k]
end

function ∂yA_over_h(i, j, k, grid, A, h)
    return ℑyᵃᶜᵃ(i, j, k, grid, ∂yᶜᶠᶜ, A)/h[i, j, k]
end

function ∂x∂xA_over_h(i, j, k, grid, A, h)
    return ∂xᶠᶜᶜ(i, j, k, grid, ∂xA_over_h, A, h)
end

function ∂y∂yA_over_h(i, j, k, grid, A, h)
    return ∂yᶜᶠᶜ(i, j, k, grid, ∂yA_over_h, A, h)
end

function jacobian_x(i, j, k, grid, fields)
    return ((∂xᶠᶜᶜ(i, j, k, grid, fields.A)       
         * ℑxyᶠᶜᵃ(i, j, k, grid, ∂y∂yA_over_h, fields.A, fields.h)) 
        - (ℑxyᶠᶜᵃ(i, j, k, grid, ∂yᶜᶠᶜ, fields.A) )
          * ∂xᶠᶜᶜ(i, j, k, grid, ∂yA_over_h, fields.A, fields.h))
end

function jacobian_y(i, j, k, grid, fields)
    return ((ℑxyᶜᶠᵃ(i, j, k, grid, ∂xᶠᶜᶜ, fields.A) 
           * ∂yᶜᶠᶜ(i, j, k, grid, ∂xA_over_h, fields.A, fields.h)) 
          - (∂yᶜᶠᶜ(i, j, k, grid, fields.A)        
          * ℑxyᶜᶠᵃ(i, j, k, grid, ∂x∂xA_over_h, fields.A, fields.h)))
end

function lorentz_force_func_x(i, j, k, grid, clock, fields)
    return (1/ℑxᶠᵃᵃ(i, j, k, grid, fields.h))*(jacobian_x(i, j, k, grid, fields))
end

function lorentz_force_func_y(i, j, k, grid, clock, fields)
    return (-1/ℑyᵃᶠᵃ(i, j, k, grid, fields.h))*(jacobian_y(i, j, k, grid, fields))
end
