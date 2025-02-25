## Bifurcation Plotting for the voting game
using Plots, LinearAlgebra, Roots

# Parameter setup
c_vals = range(0, 1, length=10000)  # Range for c
z = 0.5;  # Set z (adjust as needed)
v_equilibrium = Float64[]  # Store equilibrium values
c_equilibrium = Float64[]  # Store corresponding c values
stability = Int[]  # 1: stable, 0: unstable, 2: center points

c_comp = range(0, 1 - z - g, length=100)  # Range of c values

# Function to solve for equilibrium points v*
function equilibrium_eq(v, c, g, z)
    return v^2 * c + g * (1 - v)^2 + (z - v) * (2v^2 - 2v + 1)
end

# Jacobian function with v_c, v_g, v_z
function jac(v, c, g)
    denom = 2v^2 - 2v + 1
    v_c = v^2 / denom
    v_g = (1 - v)^2 / denom
    v_z = 1  

    J11 = -v^2 + 2c * (1 - v_c) * v - (1 - v)^2 + 2c * v_c * (1 - v)
    J12 = 2g * ((1 - v_c) * v - v_c * (1 - v))
    J21 = 2c * ((1 - v_g) * (1 - v) - v * v_g)
    J22 = -(1 - v)^2 + (1 - v_g) * 2g * (1 - v) - v^2 - 2v_g * g * v

    return [J11 J12; J21 J22]
end

# Loop over c values
for c in c_comp
    g = 1 - c - z  # 
    
    if g < 0  # Ensure g is non-negative
        continue
    end

    # Find real equilibrium points v* by solving f(v) = 0
    v_real = find_zeros(v -> equilibrium_eq(v, c, g, z), -0.1, 1.1)  #include endpoints

    # Loop over valid equilibrium points
    for v in v_real
        # Compute Jacobian matrix
        J = jac(v, c, g)
        # Compute eigenvalues
        λ = eigvals(J)
        # Determine stability
        if all(real(λ) .< 0)
            push!(stability, 1)  # Stable (solid points, blue)
        elseif all(real(λ) .== 0)  # Check for purely imaginary eigenvalues
            push!(stability, 2)  # Center points (green)
        else
            push!(stability, 0)  # Unstable (open points, red)
        end
        push!(v_equilibrium, v)
        push!(c_equilibrium, c)
    end
end

# Plot bifurcation diagram
scatter(c_equilibrium[stability .== 1], v_equilibrium[stability .== 1], 
    color=:blue, marker=:circle, label="Stable Equilibria")
scatter!(c_equilibrium[stability .== 0], v_equilibrium[stability .== 0], 
    color=:red, marker=:circle, label="Unstable Equilibria")
scatter!(c_equilibrium[stability .== 2], v_equilibrium[stability .== 2], 
    color=:green, marker=:circle, label="Center Equilibria")  # New green centers

xlabel!("c")
ylabel!("v")
title!("Bifurcation Diagram (g=1-c-z, z = $z)")
plot!(legend=:outertopright)  # Display the plot interactively
# Save the plot as a PDF
#savefig("bifurcation_diagram_z($z)_g(1-c-z).pdf")
