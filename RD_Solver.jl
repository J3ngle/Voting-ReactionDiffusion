using DifferentialEquations, Plots, LinearAlgebra, Roots, Statistics, Sundials, ColorSchemes, Unitful
@time begin
# Parameters for computations
D_c = 1e-3 #3 #Diffusion Coefficient for Consensus makers
D_g = 1e-3 #3 #Diffusion Coefficient for Gridlockers
D_z = 1e-3 #3 #Diffusion Coefficient for Zealots
D_z2 = 1e-3 #3 #Diffusion Coefficient for Zealots Party 2 
L = 10 #Length of domain 
Nx, Ny = 10, 10 #Number of discretization points in either direction
dx = L / (Nx - 1) #Chop up x equally
dy = L / (Ny - 1) #Chop up y equally
x = range(0, L, length=Nx) # X size
y = range(0, L, length=Ny) # y size 
tfinal=25.0 #Final time
X, Y = [xi for xi in x, yi in y], [yi for xi in x, yi in y]

# Gaussian shell
function gaussian(x, y, x0, y0, sigma)
    return exp.(-((x .- x0).^2 + (y .- y0).^2) ./ (2*sigma^2))
end
#Initial distribution/ conditions
N=Nx
c₀ = rand(N,N) #gaussian(X, Y, 0.5, 0.5, var) #initial distribution for Consensus makers
#c₀ = clamp.(c₀, 0, 1) #Control the bounds of initial conditions
g₀ = rand(N,N) #gaussian(X, Y, 0.5, 0.4, var) #initial distribution for Gridlockers
#g₀ = clamp.(g₀, 0, 0.05) #Control the bounds of initial conditions, we can change these around to see what will happen
z1₀ = rand(N,N) #gaussian(X, Y, 0.5, 0.5, var) #initial distribution for Zealots Party 1
z2₀ = rand(N,N) #gaussian(X, Y, 0.5, 0.5, var) #initial distribution for Zealots Party 2
#z2₀ = clamp.(z2₀, 0, 0.01) #Control the bounds of initial conditions, we can change these around to see what will happen
τ= c₀+g₀+z1₀+z2₀ 
c₀=c₀ ./ τ
g₀=g₀ ./ τ
z1₀=z1₀ ./ τ
v_0 = c₀ + g₀ + z1₀ #Initial vote on the domain
v_c₀= v_0.^2 ./(2*v_0.^2 .- 2*v_0 .+ 1) #Initial vote for Consensus makers
v_g₀= (1 .- v_0).^2 ./(2*v_0.^2 .- 2*v_0 .+ 1) #Initial vote for Gridlockers

# Pack and unpack functions for the state vector
# This is used to convert the 2D arrays into a 1D vector for the ODE solver
# and to convert it back to the 2D arrays after the ODE solver has computed the solution
function pack(c, g, z, z2, v_c, v_g)
    return vcat(vec(c), vec(g), vec(z), vec(z2), vec(v_c), vec(v_g))
end

function unpack(u)
    N = Nx * Ny
    c   = reshape(u[1:N], Nx, Ny)
    g   = reshape(u[N+1:2N], Nx, Ny)
    z   = reshape(u[2N+1:3N], Nx, Ny)
    z2  = reshape(u[3N+1:4N], Nx, Ny)
    v_c = reshape(u[4N+1:5N], Nx, Ny)
    v_g = reshape(u[5N+1:6N], Nx, Ny)
    return c, g, z, z2, v_c, v_g
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
#Construct gradient for later use
function Gradient(u, dx, dy)
    # Get size of the field
    Nx, Ny = size(u)
    #  Set up blank arrays
    grad_x = zeros(Nx, Ny)
    grad_y = zeros(Nx, Ny) 
    # Compute gradient in x direction using central difference
    for i in 2:Nx-1
        for j in 1:Ny
            grad_x[i, j] = (u[i+1, j] - u[i-1, j]) / (2dx)
        end
    end
    # Forward/backward differences at boundaries (No end points)
    grad_x[1, :] = (u[2, :] - u[1, :]) / dx
    grad_x[end, :] = (u[end, :] - u[end-1, :]) / dx
    # Compute gradient in y direction using central difference
    for i in 1:Nx
        for j in 2:Ny-1
            grad_y[i, j] = (u[i, j+1] - u[i, j-1]) / (2dy)
        end
    end
    # Forward/backward differences at boundaries (No end points)
    grad_y[:, 1] = (u[:, 2] - u[:, 1]) / dy
    grad_y[:, end] = (u[:, end] - u[:, end-1]) / dy
    #Store in both x and y directions
    return [grad_x, grad_y]
end



# Fitness functions
Fitness_c(v) = 4 * (v .- 0.5).^2 #Strategy fitness for Consensus makers
Fitness_g(v) = 1 .- 4*(v .- 0.5).^2 #Strategy fitness for Gridlockers
Fitness_z1(v) = (v).^2 #Strategy fitness for Zealots party 1
Fitness_z2(v) = 1 .- (v).^2 #Strategy fitness for Zealots party 2
#Set up equilibrium equation

#not used 
function equilibrium_eq(v, c, g, z)
    return v.^2 .* c .+ g .* (1 .- v).^2 .+ (z .- v) .* (2 .* v.^2 - 2 .* v .+ 1)
end

# Initial condition
u0 = pack(c₀, g₀, z1₀, v_c₀, v_g₀)
du0 = zeros(size(u0))
time = (0.0, tfinal)

# # Define the PDE system using v as a fixed spatial array
# function pdes!(du, u, p, t)
#     c, g, z, v_c, v_g = unpack(u)
#     # Fitness values are matrices now
#     v = c .* v_c + g .* v_g + z
#     F_c = Fitness_c(v)
#     F_g = Fitness_g(v)
#     F_z = Fitness_z1(v)
#     F_z2= Fitness_z2(v)
#     # Reaction Diffusion equation with Replicator Equation
#     du_c = D_c * laplacian(c) + (c .* g .* (F_c .- F_g) + c .* z .* (F_c .- F_z) + c .* z2 .* (F_c .- F_z2))
#     du_g = D_g * laplacian(g) + (g .* c .* (F_g .- F_c) + g .* z .* (F_g .- F_z)+ g .* z2 .* (F_g .- F_z2))
#     du_z = D_z * laplacian(z) + (z .* c .* (F_z .- F_c) + z .* g .* (F_z .- F_g))
#     f(v) =equilibrium_eq(v, c, g, z)
#     denominator = 2 .* v.^2 .- 2 .* v .+ 1
#     v_c = f(v).^2 ./ denominator #((1 .-v_c).*v.^2+v_c.*(1 .-v).^2)
#     v_g = (1 .- f(v)).^2 ./ denominator #(1-(v_g).*v.^2+(1 .-v_g).*(1 .-v).^2)
#     du .= pack(du_c, du_g, du_z, v_c, v_g)
# end
# # # Solve the PDE
# prob = ODEProblem(pdes!, u0, tspan)
# sol = solve(prob, Rodas5(), abstol=1e-6,saveat=tfinal)

end

function DAE!(du, u, p, t)
    c, g, z, z2, v_c, v_g = unpack(u)
    # du_c, du_g, du_z, du_v_c, du_v_g = unpack(du)
    v = c .* v_c .+ g .* v_g .+ z #vote for party 1

    F_c = Fitness_c(v)
    F_g = Fitness_g(v)
    F_z = Fitness_z1(v)
    F_z2 = Fitness_z2(v)
    # Partial Differential equations
    du_c = D_c * laplacian(c) + (c .* g .* (F_c .- F_g) + c .* z .* (F_c .- F_z)+c .* z2 .* (F_c .- F_z2)) 
    du_g = D_g * laplacian(g) + (g .* c .* (F_g .- F_c) + g .* z .* (F_g .- F_z)+ g .* z2 .* (F_g .- F_z2)) 
    du_z = D_z * laplacian(z) + (z .* c .* (F_z .- F_c) + z .* g .* (F_z .- F_g))
    du_z2 = D_z2 * laplacian(z2) + (z2 .* c .* (F_z2 .- F_c) + z .* g .* (F_z2 .- F_g))
    # Algebraic equations
    du_v_c = (1 .- v_c) .* v.^2 .- v_c .* (1 .- v).^2
    du_v_g = (1 .- v_g) .* (1 .- v).^2 .- v_g .* v.^2
    # du_v_c = clamp.(du_v_c, 0, 1)
    # du_v_g = clamp.(du_v_g, 0, 1)
    du .= pack(du_c, du_g, du_z, du_z2, du_v_c, du_v_g)
end

# Mass matrix: 1 for c,g,z , 0 for v_c,v_g 
# function mass_matrix(u, p, t)
#M = diagm(vcat(ones(3*Nx*Ny), zeros(2*Nx*Ny)))
M = diagm(vcat(ones(6*Nx*Ny)))

u0 = pack(c₀, g₀, z1₀, z2₀, v_c₀, v_g₀)
du0 = zeros(size(u0))
time = (0.0, tfinal)

DAEfunc = ODEFunction(DAE!, mass_matrix = M)
prob = ODEProblem(DAEfunc, u0, time)
sol = solve(prob, RadauIIA5(), saveat=0.01, reltol=1e-10, abstol=1e-10)

# HEATMAPS: Plot results at final time 
fontsize=12
c, g, z, z2, v_c, v_g = unpack(sol[end]) #computations from the end of the simulation, we could pull these at any other times
v = c .* v_c .+ g .* v_g .+ z #Compute v at the end
pop = c .+ g .+ z .+ z2 #Compute population at the end
clims = (0, 1) #Color limits for heatmaps
p1 = heatmap(x, y, c', title="c(x,y)", xlabel="x", ylabel="y", aspect_ratio=1,colorbar=false, clims=clims)
p2 = heatmap(x, y, g', title="g(x,y)", xlabel="x", ylabel="y", aspect_ratio=1,colorbar=false, clims=clims)
p3 = heatmap(x, y, z', title="z(x,y)", xlabel="x", ylabel="y", aspect_ratio=1,colorbar=false, clims=clims)
p4 = heatmap(x, y, z2', title="z2(x,y)", xlabel="x", ylabel="y", aspect_ratio=1,colorbar=false, clims=clims)
p5 = heatmap(x, y, pop', title="Population", xlabel="x", ylabel="y", aspect_ratio=1,colorbar=true)
p6 = heatmap(x, y, v', title="v(x,y)", xlabel="x", ylabel="y", aspect_ratio=1,colorbar=false, color=:balance)
heatmap_figure = plot(p1, p2, p3, p4, p5, p6, layout=(3,3), size=(1600, 1600),colorbar=true, titlefontsize=fontsize, guidefontsize=fontsize, tickfontsize=fontsize, plot_title="Solutions at final time $tfinal, D=$D ")
display(heatmap_figure)
#savefig("Heatmap_D_0.001_Finaltime=$tfinal.pdf")

# TIME SERIES: Compute averages over the domain at each time step
time_steps = sol.t
average_c = [mean(unpack(sol[i])[1]) for i in 1:length(time_steps)]
average_g = [mean(unpack(sol[i])[2]) for i in 1:length(time_steps)]
average_z = [mean(unpack(sol[i])[3]) for i in 1:length(time_steps)]
average_z2 = [mean(unpack(sol[i])[4]) for i in 1:length(time_steps)]
ts_max_pop = [maximum(unpack(sol[i])[1]) + maximum(unpack(sol[i])[2]) + maximum(unpack(sol[i])[3]) + maximum(unpack(sol[i])[4]) for i in 1:length(time_steps)]
average_v = [mean(unpack(sol[i])[5]) .* mean(unpack(sol[i])[1])  + mean(unpack(sol[i])[2]) .* mean(unpack(sol[i])[6]) + mean(unpack(sol[i])[3]) + mean(unpack(sol[i])[4]) for i in 1:length(time_steps)]
# Above computes c*v_c + g*v_g + z at each time step
# Plot averages
time_series = plot(time_steps, average_c, label="Mean Consensus Makers", xlabel="Time", ylabel="Mean", title="Time Series averages of c, g, z, v ",lw=3,legend=:outertop)
plot!(time_steps, average_g, label="Mean Gridlockers",lw=3)
plot!(time_steps, average_z, label="Mean Zealots of party 1",lw=3)
plot!(time_steps, average_z2, label="Mean Zealots of party 2",lw=3)
plot!(time_steps, average_v, label="Mean Vote for party 1",lw=3)
plot!(time_steps, ts_max_pop, label="Max Population",lw=3)
display(time_series)
#savefig("TS_D_0.001_Finaltime=$tfinal.pdf")


#Sanity check: Plot the average v_c and v_g 
mean_vc = [mean(unpack(sol[i])[4]) for i in 1:length(time_steps)]
mean_vg = [mean(unpack(sol[i])[5]) for i in 1:length(time_steps)]
min_vc = [minimum(unpack(sol[i])[4]) for i in 1:length(time_steps)]
min_vg = [minimum(unpack(sol[i])[5]) for i in 1:length(time_steps)]
max_vc = [maximum(unpack(sol[i])[4]) for i in 1:length(time_steps)]
max_vg = [maximum(unpack(sol[i])[5]) for i in 1:length(time_steps)]
SanityCheck=plot(time_steps, mean_vc, label="Mean v_c", xlabel="Time", ylabel="Mean", lw=3, legend=:outertopright, title="Mean v_c and v_g over time, Sanity Check")
plot!(time_steps, mean_vg, label="Mean v_g", lw=3)
plot!(time_steps, min_vc, label="Min v_c", lw=3)
plot!(time_steps, min_vg, label="Min v_g", lw=3)
plot!(time_steps, max_vc, label="Max v_c", lw=3)
plot!(time_steps, max_vg, label="Max v_g", lw=3)
display(SanityCheck)
#savefig("MeanTS(SanityCheck)_D_0.01_Finaltime=$tfinal.pdf")