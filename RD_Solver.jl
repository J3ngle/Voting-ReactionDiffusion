using DifferentialEquations, Plots, LinearAlgebra, Roots, Statistics, Sundials
@time begin
# Parameters for comp.
D = 1 #Diffusion Coefficient
L = 10 #Discretization steps 
Nx, Ny = 4, 4 #Number of discretization points in either direction
dx = L / (Nx - 1) #Chop up x
dy = L / (Ny - 1) #Chop up y
x = range(0, L, length=Nx) # X size
y = range(0, L, length=Ny) # y size 
tfinal=100.0 #Final time
X, Y = [xi for xi in x, yi in y], [yi for xi in x, yi in y]

# Gaussian shell
function gaussian(x, y, x0, y0, sigma)
    return exp.(-((x .- x0).^2 + (y .- y0).^2) ./ (2*sigma^2))
end
#Initial distribution/ conditions
N=Nx
c₀ = rand(N,N) #gaussian(X, Y, 0.5, 0.5, var) #initial distribution for Consensus makers
g₀ = rand(N,N) #gaussian(X, Y, 0.5, 0.4, var) #initial distribution for Gridlockers
z1₀ = rand(N,N) #gaussian(X, Y, 0.5, 0.5, var) #initial distribution for Zealots Party 1
z2₀ = rand(N,N) #gaussian(X, Y, 0.5, 0.5, var) #initial distribution for Zealots Party 2
τ= c₀+g₀+z1₀
c₀=c₀ ./ τ
g₀=g₀ ./ τ
z1₀=z1₀ ./ τ
v_c₀= rand(N,N)
v_g₀= rand(N,N)
function pack(c, g, z, v_c, v_g)
    return vcat(vec(c), vec(g), vec(z), vec(v_c), vec(v_g))
end

function unpack(u)
    N = Nx * Ny
    c   = reshape(u[1:N], Nx, Ny)
    g   = reshape(u[N+1:2N], Nx, Ny)
    z   = reshape(u[2N+1:3N], Nx, Ny)
    v_c = reshape(u[3N+1:4N], Nx, Ny)
    v_g = reshape(u[4N+1:5N], Nx, Ny)
    return c, g, z, v_c, v_g
end

#Construct laplacian in 2D
function laplacian(U)
    L = similar(U)
    # Wrap around for periodic boundary conditions
    for i in 1:Nx, j in 1:Ny
        ip1 = mod(i, Nx) + 1  
        im1 = mod(i - 2, Nx) + 1
        jp1 = mod(j, Ny) + 1
        jm1 = mod(j - 2, Ny) + 1
        L[i, j] = (U[ip1, j] + U[im1, j] + U[i, jp1] + U[i, jm1] - 4U[i, j]) / dx^2
    end
    return L
end

# Fitness functions
Fitness_c(v) = 4 * (v .- 0.5).^2 #Strategy fitness for Consensus makers
Fitness_g(v) = 1 .- 4*(v .- 0.5).^2 #Strategy fitness for Gridlockers
Fitness_z1(v) = (v).^2 #Strategy fitness for Zealots party 1
Fitness_z2(v) = 1 .- (v).^2 #Strategy fitness for Zealots party 2
#Set up equilibrium equation

function equilibrium_eq(v, c, g, z)
    return v.^2 .* c .+ g .* (1 .- v).^2 .+ (z .- v) .* (2 .* v.^2 - 2 .* v .+ 1)
end

# Initial condition
u0 = pack(c₀, g₀, z1₀, v_c₀, v_g₀)
du0 = zeros(size(u0))
time = (0.0, tfinal)

# Define the PDE system using v as a fixed spatial array
function pdes!(du, u, p, t)
    c, g, z, v_c, v_g = unpack(u)
    # Fitness values are matrices now
    v = c .* v_c + g .* v_g + z
    F_c = Fitness_c(v)
    F_g = Fitness_g(v)
    F_z = Fitness_z1(v)
    # Reaction Diffusion equation with Replicator Equation
    du_c = D * laplacian(c) + (c .* g .* (F_c .- F_g) + c .* z .* (F_c .- F_z))
    du_g = D * laplacian(g) + (g .* c .* (F_g .- F_c) + g .* z .* (F_g .- F_z))
    du_z = D * laplacian(z) + (z .* c .* (F_z .- F_c) + z .* g .* (F_z .- F_g))
    f(v) =equilibrium_eq(v, c, g, z)
    denominator = 2 .* v.^2 .- 2 .* v .+ 1
    v_c = f(v).^2 ./ denominator #((1 .-v_c).*v.^2+v_c.*(1 .-v).^2)
    v_g = (1 .- f(v)).^2 ./ denominator #(1-(v_g).*v.^2+(1 .-v_g).*(1 .-v).^2)
    du .= pack(du_c, du_g, du_z, v_c, v_g)
end
# # # Solve the PDE
# prob = ODEProblem(pdes!, u0, tspan)
# sol = solve(prob, Rodas5(), abstol=1e-6,saveat=tfinal)

end

function DAE!(du, u, p, t)
    c, g, z, v_c, v_g = unpack(u)
    # du_c, du_g, du_z, du_v_c, du_v_g = unpack(du)
    v = c .* v_c + g .* v_g + z

    F_c = Fitness_c(v)
    F_g = Fitness_g(v)
    F_z = Fitness_z1(v)

    # Partial Differential equations
    du_c = D * laplacian(c) + (c .* g .* (F_c .- F_g) + c .* z .* (F_c .- F_z)) 
    du_g = D * laplacian(g) + (g .* c .* (F_g .- F_c) + g .* z .* (F_g .- F_z)) 
    du_z = D * laplacian(z) + (z .* c .* (F_z .- F_c) + z .* g .* (F_z .- F_g))
    # Algebraic equations
    du_v_c = (1 .- v_c).*v.^2 .- v_c*(1 .- v).^2
    du_v_g = (1 .- v_g).*(1 .- v).^2 .- v_g.*v.^2
    du .= pack(du_c, du_g, du_z, du_v_c, du_v_g)
end

# Mass matrix: 1 for c,g,z , 0 for v_c,v_g 
# function mass_matrix(u, p, t)
 M = diagm(vcat(ones(3*Nx*Ny), zeros(2*Nx*Ny)))
# M = diagm(vcat(ones(5*Nx*Ny)))


# end

u0 = pack(c₀, g₀, z1₀, v_c₀, v_g₀)
du0 = zeros(size(u0))
time = (0.0, tfinal)
# Resize to match the number of variables
N = Nx * Ny
# Store our differential variables
differential_vars = vcat(trues(3N), falses(2N))
## Solve the DAE system
# f = ODEFunction(system)
# prob = ODEProblem(system, u0, tspan, mass_matrix = mass_matrix)
# sol = solve(prob, Rodas5(), abstol=1e-6,saveat=tfinal)

DAEfunc = ODEFunction(DAE!, mass_matrix = M)
prob = ODEProblem(DAEfunc, u0, time)
sol = solve(prob, RadauIIA5(), saveat=1, reltol=1e-10, abstol=1e-10)

#Plot results at final time 
c, g, z, v_c, v_g = unpack(sol[end]) #computations from the end of the simulation, we could pull these at any other times
v = c .* v_c + g .* v_g + z #Compute v at the end
p1 = heatmap(x, y, c', title="c(x,y)", xlabel="x", ylabel="y", aspect_ratio=1)
p2 = heatmap(x, y, g', title="g(x,y)", xlabel="x", ylabel="y", aspect_ratio=1)
p3 = heatmap(x, y, z', title="z(x,y)", xlabel="x", ylabel="y", aspect_ratio=1)
p4 = heatmap(x, y, v', title="v(x,y)", xlabel="x", ylabel="y", aspect_ratio=1)
heatmap_figure = plot(p1, p2, p3, p4, layout=(1,4), size=(1600, 400), plot_title="Solutions at final time $tfinal, D=$D ")
display(heatmap_figure)

# Compute averages over the domain at each time step
time_steps = sol.t
average_c = [mean(unpack(sol[i])[1]) for i in 1:length(time_steps)]
average_g = [mean(unpack(sol[i])[2]) for i in 1:length(time_steps)]
average_z = [mean(unpack(sol[i])[3]) for i in 1:length(time_steps)]
# This needs to be fixed, v_c is too big 
average_v = [mean(unpack(sol[i])[4]) .* mean(unpack(sol[i])[1])  + mean(unpack(sol[i])[2]) .* mean(unpack(sol[i])[5]) + mean(unpack(sol[i])[3]) for i in 1:length(time_steps)]
# Above computes c*v_c + g*v_g + z at each time step
# Plot averages
time_series = plot(time_steps, average_c, label="Mean Consensus Makers on entire Domain", xlabel="Time", ylabel="Mean", title="Time Series averages of c, g, z, v ",lw=3,legend=:topright)
plot!(time_steps, average_g, label="Mean Gridlockers on entire Domain",lw=3)
plot!(time_steps, average_z, label="Mean Zealots on entire Domain",lw=3)
plot!(time_steps, average_v, label="Mean Vote on entire Domain",lw=3)
display(time_series)
#savefig("TS_D_0.1_Finaltime=$tfinal.pdf")
#savefig("Heatmap_Periodic_D_0.1_Finaltime=$tfinal.pdf")

#Sanity check: Plot the average v_c and v_g 
mean_vc = [mean(unpack(sol[i])[4]) for i in 1:length(time_steps)]
mean_vg = [mean(unpack(sol[i])[5]) for i in 1:length(time_steps)]

plot(time_steps, mean_vc, label="Mean v_c", xlabel="Time", ylabel="Mean", lw=3, legend=:topright, title="Mean v_c and v_g over time")
plot!(time_steps, mean_vg, label="Mean v_g", lw=3)
display(current())


