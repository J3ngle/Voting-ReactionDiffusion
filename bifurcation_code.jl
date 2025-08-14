## Bifurcation Plotting for the voting game
using Plots, LinearAlgebra, Roots

# Parameter setup
c_vals = range(0, 1, length=10000)  # Range for c
z = 0.2;  # Set z (adjust as needed)
v_equilibrium = Float64[]  # Store equilibrium values
c_equilibrium = Float64[]  # Store corresponding c values
stability = Int[]  # 1: stable, 0: unstable, 2: center points
N=1000
c_comp = range(0, 1 - z, length=N)  # Range of c values

# Function to solve for equilibrium points v*
function equilibrium_eq(v, c, g, z)
    return v^2 * c + g * (1 - v)^2 + (z - v) * (2v^2 - 2v + 1)
end

# Jacobian function with v_c, v_g, v_z
function jac(v, c, g)
    denom = 2v^2 - 2v + 1
    v_c = v^2 / denom
    v_g = (1 - v)^2 / denom

    J11 = -v^2 + 2*c*(1-v_c)*v - (1-v)^2 + 2*c*v_c*(1-v)
    J12 = 2*g*((1-v_c)*v + v_c*(1-v))
    J21 = -2*c*((1-v_g)*(1-v) + v_g*v)
    J22 = -(1-v)^2 - 2*g*(1-v_g)*(1-v) - v^2 - 2*g*v_g*v

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

# Group and sort by stability
stable_idx   = findall(stability .== 1)
unstable_idx = findall(stability .== 0)
center_idx   = findall(stability .== 2)

stable_c = c_equilibrium[stable_idx]
stable_v = v_equilibrium[stable_idx]
unstable_c = c_equilibrium[unstable_idx]
unstable_v = v_equilibrium[unstable_idx]
center_c = c_equilibrium[center_idx]
center_v = v_equilibrium[center_idx]

# Sort each group by c for proper line plotting
stable_order   = sortperm(stable_c)
unstable_order = sortperm(unstable_c)
center_order   = sortperm(center_c)

stable_c_sorted   = stable_c[stable_order]
stable_v_sorted   = stable_v[stable_order]
unstable_c_sorted = unstable_c[unstable_order]
unstable_v_sorted = unstable_v[unstable_order]
center_c_sorted   = center_c[center_order]
center_v_sorted   = center_v[center_order]

# Get unique c values for each group
unique_c_stable   = unique(stable_c_sorted)
unique_c_unstable = unique(unstable_c_sorted)
unique_c_center   = unique(center_c_sorted)

# For each unique c, get min/max v for stable, unstable, center
min_v_stable = [minimum(stable_v_sorted[stable_c_sorted .== c]) for c in unique_c_stable]
max_v_stable = [maximum(stable_v_sorted[stable_c_sorted .== c]) for c in unique_c_stable]

min_v_unstable = [minimum(unstable_v_sorted[unstable_c_sorted .== c]) for c in unique_c_unstable]
max_v_unstable = [maximum(unstable_v_sorted[unstable_c_sorted .== c]) for c in unique_c_unstable]

min_v_center = [minimum(center_v_sorted[center_c_sorted .== c]) for c in unique_c_center]
max_v_center = [maximum(center_v_sorted[center_c_sorted .== c]) for c in unique_c_center]



# Find the smallest c where unstable equilibrium exists
if !isempty(unique_c_unstable)
    c_unstable_min = minimum(unique_c_unstable)
    # Filter stable c and min_v_stable for c ≥ c_unstable_min
    stable_mask = unique_c_stable .>= c_unstable_min
    filtered_c_stable = unique_c_stable[stable_mask]
    filtered_min_v_stable = min_v_stable[stable_mask]
else
    filtered_c_stable = Float64[]
    filtered_min_v_stable = Float64[]
end

# Plotting 
plot(filtered_c_stable, filtered_min_v_stable, color=:blue, lw=10, label="  Stable Equilibria")#,xlims=(0, 1.02)
plot!(unique_c_unstable, min_v_unstable, color=:red, lw=10, label="  Unstable Equilibria")
plot!(unique_c_stable, max_v_stable, color=:blue, lw=10, label= false)

xlabel!("C")
ylabel!("V")
plot!(legend=:topleft, xlabelfontsize=20,tickfontsize=16, ylabelfontsize=20, titlefontsize=24, legendfontsize=20) #,xlims=(0, 1.02)
savefig("2Bifurcation_diagram_z($z)_g(1-c-z).pdf")
