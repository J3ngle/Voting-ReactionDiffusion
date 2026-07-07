using Plots

# Parameter setup
c_vals = range(0, 1, length=10000)  # Range for c
z = 0  # Set z (adjust as needed)
v_equilibrium = Float64[]  # Store equilibrium values
c_equilibrium = Float64[]  # Store c values
stability = Int[]  # Store stability (1: stable, 0: unstable)
c_comp = range(0, 1 - z, length=100)

# Loop over c values
for c in c_comp
    g = 0  # Fixed g value
    v_real = [0.5]  # Always include v = 0.5
    
    # Check conditions for additional equilibrium points
    if (4 * c - 3) >= 0
        v1 = 0.5  
        v2 = 0.5 * (1 - sqrt(4 * c - 3))
        v3 = 0.5 * (1 + sqrt(4 * c - 3))
        append!(v_real, [v1, v2, v3])
    end
    if c <=.5 
        v4=0
        append!([v4])
    end
    # Loop over valid equilibrium points
    for v in v_real
        # Compute stability: take derivative
        vdot = 2 * c * v - 2 * g * (1 - v) + (z - v) * (4 * v - 2) - (2 * v^2 - 2 * v + 1)

        if vdot < 0
            push!(stability, 1)  # Stable (solid points)
        else
            push!(stability, 0)  # Unstable (open points)
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

xlabel!("c")
ylabel!("v")
title!("Bifurcation Diagram z = $z, g = 0")
plot!(legend=:topright)

# Save figure as PDF
savefig("bifurcation_diagram.pdf")

println("Plot saved as 'bifurcation_diagram.pdf'.")
