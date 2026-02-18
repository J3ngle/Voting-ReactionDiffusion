#VariancePlots
using DifferentialEquations, Plots, LinearAlgebra, Roots, Statistics, Sundials, ColorSchemes, SparseArrays
@time begin

    iter = 5
    D_i_array = range(0.001, 0.1, length=iter) #Diffusion coefficient for all types, we can change this around to see what will happen
    averagevar = zeros(iter) #Variance
    maxvar = zeros(iter) #Max variance
    var_array = zeros(iter) #Array to store variance for each iteration, we can use this to plot the distribution of variances across the domain for different diffusion coefficients
for i in 1:iter
    local D_i, D_c, D_g, D_z, D_z2, m_c, m_g, m_z, m_z2, V, λ, b, k, s, L, Nx, Ny, dx, dy, x, y, tfinal, X, Y, N, c₀, g₀, z1₀, z2₀, τ, v_0, v_c₀, v_g₀, u0, du0, tspan, sol, c, g, z, z2, v_c, v_g, population, total_population, heatmap_population, v
    
D_i = D_i_array[i]
# Parameters for computations
D_c = D_i #0.1#1e-0 #Diffusion Coefficient for Consensus makers
D_g = D_i #1e-13 #3 #Diffusion Coefficient for Gridlockers
D_z = D_i #0.1#1e-0 #Diffusion Coefficient for Zealots
D_z2 = D_i #0.15 #1e-13 #3 #Diffusion Coefficient for Zealots Party 2
m_c = 0 #1e-10 # #Migration rate for Consensus makers
m_g =  0 # #Migration rate for Gridlockers
m_z = 0 #1e-0 # #Migration rate for Zealots Party 1
m_z2 = 0 # #Migration rate for Zealots Party 2
V= 1 #Social Imitation
λ= 0 #Economic preference
b=0 #public good benefit
k=0 #public good cost
s= 0 #Spillovers
L = 10 #Length of domain    
Nx, Ny = 10, 10 #Number of discretization points in either direction
dx = L / (Nx - 1) #Chop up x equally
dy = L / (Ny - 1) #Chop up y equally
x = range(0, L, length=Nx) # X size
y = range(0, L, length=Ny) # y size 
tfinal=25.0 #Final time
X, Y = [xi for xi in x, yi in y], [yi for xi in x, yi in y]

#Initial distribution/ conditions
N=Nx
c₀ = rand(N,N) #gaussian(X, Y, 0.5, 0.5, var) #initial distribution for Consensus makers
#c₀ = clamp.(c₀, 0, .6) #Control the bounds of initial conditions
g₀ = rand(N,N) #gaussian(X, Y, 0.5, 0.4, var) #initial distribution for Gridlockers
#g₀ = clamp.(g₀, 0, 0.15) #Control the bounds of initial conditions, we can change these around to see what will happen
z1₀ = rand(N,N) #gaussian(X, Y, 0.5, 0.5, var) #initial distribution for Zealots Party 1
#z1₀ = clamp.(z1₀, 0, 0.15) #Control the bounds of initial conditions, we can change these around to see what will happen
z2₀ = rand(N,N) #gaussian(X, Y, 0.5, 0.5, var) #initial distribution for Zealots Party 2
#z2₀ = clamp.(z2₀, 0, 0.1) #Control the bounds of initial conditions, we can change these around to see what will happen
τ= c₀ .+ g₀ .+ z1₀ .+ z2₀
c₀=c₀ ./ τ #Normalize initial conditions
g₀=g₀ ./ τ
z1₀=z1₀ ./ τ
z2₀=z2₀ ./ τ
v_0 = c₀ .+ 0.5*g₀ .+ z1₀ #Initial vote on the domain
v_c₀= v_0.^2 ./(2 .* v_0.^2 .- 2 .* v_0 .+ 1) #Initial vote for Consensus makers
v_g₀= (1 .- v_0).^2 ./ (2 .*v_0.^2 .- 2 .* v_0 .+ 1) #Initial vote for Gridlockers

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
    Nx, Ny = size(u)
    T = eltype(u)  # Get the element type (Float64 or Dual)
    grad_x = zeros(T, Nx, Ny)
    grad_y = zeros(T, Nx, Ny) 
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
    # Forward/backward differences at boundaries
    grad_y[:, 1] = (u[:, 2] - u[:, 1]) / dy
    grad_y[:, end] = (u[:, end] - u[:, end-1]) / dy
    #Store in both x and y directions
    return [grad_x, grad_y]
end

function spillover_integral(sol, Nx, Ny, dx, dy)
    spillover_v = zeros(length(sol))
    for i in 1:length(sol)
        c, g, z, z2, v_c, v_g = unpack(sol[i])
        v = c .* v_c .+ g .* v_g .+ z
        integral = sum(v) * dx * dy #Trapezoid rule
        spillover_v[i] = integral ./ (Nx * Ny) #Average over the domain
    end
    return spillover_v
end


# Fitness functions
Fitness_c(v) =  (1 .+ cos.(2 .* pi .* v)) ./ 2 #4 .* (v .- 0.5).^2 #Strategy fitness for Consensus makers
Fitness_g(v) = (1 .- cos.(2 .* pi .* v)) ./ 2#1 .- 4 .* (v .- 0.5).^2 #Strategy fitness for Gridlockers
Fitness_z1(v) = (1 .- cos.(pi .* v)) ./2 #(v).^2 #Strategy fitness for Zealots party 1
Fitness_z2(v) = (1 .+ cos.(pi .* v)) ./2 #(1 .- (v)).^2 #Strategy fitness for Zealots party 2

#Economic Utility functions 

Utility_c(v, spillover_v) = λ .* ((1-s)*b .* v .- k .* v .+ s*b .* spillover_v) .* v .+ (1-λ).*Fitness_c(v)
Utility_g(v, spillover_v) = λ .* ((1-s)*b .* v .- k .* v .+ s*b .* spillover_v) .* v .+ (1-λ).*Fitness_g(v)
Utility_z1(v, spillover_v) = λ .* ((1-s)*b .* v .- k .* v .+ s*b .* spillover_v) .* v .+ (1-λ).*Fitness_z1(v)
Utility_z2(v, spillover_v) = λ .* ((1-s)*b .* v .- k .* v .+ s*b .* spillover_v) .* v .+ (1-λ).*Fitness_z2(v)



# Initial condition
u0 = pack(c₀, g₀, z1₀, z2₀, v_c₀, v_g₀)
du0 = zeros(size(u0))
tspan = (0.0, tfinal)



function DAE!(du, u, p, t)
    c, g, z, z2, v_c, v_g = unpack(u)
    # du_c, du_g, du_z, du_v_c, du_v_g = unpack(du)
    v = c .* v_c .+ g .* v_g .+ z #vote for party 1

    F_c = Fitness_c(v)
    F_g = Fitness_g(v)
    F_z = Fitness_z1(v)
    F_z2 = Fitness_z2(v)

    spillover_v = sum(v) * dx * dy / (Nx * Ny)
    
    u_c = Utility_c(v, spillover_v)
    u_g = Utility_g(v, spillover_v)
    u_z1 = Utility_z1(v, spillover_v)
    u_z2 = Utility_z2(v, spillover_v)
    # Gradients of utilities in x and y directions
    grad_uc_x =  Gradient(u_c, dx, dy)[1] 
    grad_uc_y =  Gradient(u_c, dx, dy)[2]
    grad_ug_x =  Gradient(u_g, dx, dy)[1]
    grad_ug_y =  Gradient(u_g, dx, dy)[2]
    grad_uz1_x = Gradient(u_z1, dx, dy)[1]
    grad_uz1_y = Gradient(u_z1, dx, dy)[2]
    grad_uz2_x = Gradient(u_z2, dx, dy)[1]
    grad_uz2_y = Gradient(u_z2, dx, dy)[2]
    #Compute divergence of c grad u_c
    div_c_grad_uc_x= Gradient(c .* grad_uc_x, dx, dy)[1] #x direction
    div_c_grad_uc_y= Gradient(c .* grad_uc_y, dx, dy)[2] #y direction
    div_c_grad_uc= div_c_grad_uc_x .+ div_c_grad_uc_y #total divergence
     #Compute divergence of g grad u_g
    div_g_grad_uc_x= Gradient(g .* grad_ug_x, dx, dy)[1] #x direction
    div_g_grad_uc_y= Gradient(g .* grad_ug_y, dx, dy)[2] #y direction
    div_g_grad_ug= div_g_grad_uc_x .+ div_g_grad_uc_y #total divergence
     #Compute divergence of z1 grad u_z1
    div_z1_grad_uz1_x= Gradient(z .* grad_uz1_x, dx, dy)[1] #x direction
    div_z1_grad_uz1_y= Gradient(z .* grad_uz1_y, dx, dy)[2] #y direction
    div_z1_grad_uz1= div_z1_grad_uz1_x .+ div_z1_grad_uz1_y #total divergence
    #Compute divergence of z2 grad u_z2
    div_z2_grad_uz2_x= Gradient(z2 .* grad_uz2_x, dx, dy)[1] #x direction
    div_z2_grad_uz2_y= Gradient(z2 .* grad_uz2_y, dx, dy)[2] #y direction
    div_z2_grad_uz2= div_z2_grad_uz2_x .+ div_z2_grad_uz2_y #total divergence
    # # Partial Differential equations
    du_c = D_c .* laplacian(c) .-m_c .*div_c_grad_uc .+ V.* (c .* g .* (F_c .- F_g) + c .* z .* (F_c .- F_z) + c .* z2 .* (F_c .- F_z2))
    du_g = D_g .* laplacian(g) .-m_g .*div_g_grad_ug .+ V.* (g .* c .* (F_g .- F_c) + g .* z .* (F_g .- F_z) + g .* z2 .* (F_g .- F_z2))
    du_z = D_z .* laplacian(z) .-m_z .*div_z1_grad_uz1 .+ V.* (z .* c .* (F_z .- F_c) + z .* g .* (F_z .- F_g))
    du_z2 = D_z2 .* laplacian(z2) .-m_z2 .*div_z2_grad_uz2 .+ V.* (z2 .* c .* (F_z2 .- F_c) + z2 .* g .* (F_z2 .- F_g))
    # Partial Differential equations
    # du_c = D_c .* laplacian(c) .-m_c .*((Gradient(c, dx, dy)[1] .+ Gradient(c, dx, dy)[2]).* (grad_uc_x.+grad_uc_y).+c.*laplacian(u_c)) .+ V.* (c .* g .* (F_c .- F_g) + c .* z .* (F_c .- F_z) + c .* z2 .* (F_c .- F_z2))
    # du_g = D_g .* laplacian(g) .-m_g .*((Gradient(g, dx, dy)[1] .+ Gradient(g, dx, dy)[2]).* (grad_ug_x.+grad_ug_y).+c.*laplacian(u_g)) .+ V.* (g .* c .* (F_g .- F_c) + g .* z .* (F_g .- F_z) + g .* z2 .* (F_g .- F_z2))
    # du_z = D_z .* laplacian(z) .-m_z .*((Gradient(z, dx, dy)[1] .+ Gradient(z, dx, dy)[2]).* (grad_uz1_x.+grad_uz1_y).+c.*laplacian(u_z1)) .+ V.* (z .* c .* (F_z .- F_c) + z .* g .* (F_z .- F_g))
    # du_z2 = D_z2 .* laplacian(z2) .-m_z2 .*((Gradient(z2, dx, dy)[1] .+ Gradient(z2, dx, dy)[2]).* (grad_uz2_x.+grad_uz2_y).+c.*laplacian(u_z2)) .+ V.* (z2 .* c .* (F_z2 .- F_c) + z2 .* g .* (F_z2 .- F_g))
    # Algebraic equations
    du_v_c = (1 .- v_c) .* v.^2 .- v_c .* (1 .- v).^2
    du_v_g = (1 .- v_g) .* (1 .- v).^2 .- v_g .* v.^2
    # du_v_c = clamp.(du_v_c, 0, 1)
    # du_v_g = clamp.(du_v_g, 0, 1)
    du .= pack(du_c, du_g, du_z, du_z2, du_v_c, du_v_g)
end

# Mass matrix: 1 for c,g,z , 0 for v_c,v_g 
# function mass_matrix(u, p, t)
# M = diagm(vcat(ones(4*Nx*Ny), zeros(2*Nx*Ny)))
# M = diagm(vcat(ones(6*Nx*Ny))) #Used for quick comps
M = spdiagm(0 => ones(6*Nx*Ny)) #Need to use for large final solutions, takes longer to compute but better for memory
u0 = pack(c₀, g₀, z1₀, z2₀, v_c₀, v_g₀)
du0 = zeros(size(u0))

DAEfunc = ODEFunction(DAE!, mass_matrix = M)
prob = ODEProblem(DAEfunc, u0, tspan)
sol = solve(prob, RadauIIA5(), saveat=0.01, reltol=1e-12, abstol=1e-12) #Different solvers:RadauIIA5,Rodas5P,Rodas4,ROS34PW2,ROS34PW3,Trapezoid

c, g, z, z2, v_c, v_g = unpack(sol[end]) #computations from the end of the simulation, we could pull these at any other times
population = c .+ g .+ z .+ z2 #Compute population, this is a matrix
total_population = sum(c) .+ sum(g) .+ sum(z) .+ sum(z2) #Compute population at the end, this is a scalar
c=c ./ population #Normalize c
g=g ./ population #Normalize g
z=z ./ population #Normalize z
z2=z2 ./ population #Normalize z2
heatmap_population = population  ./ maximum(population) #Normalize population
v = (c .* v_c .+ g .* v_g .+ z) #Compute v at the end
compute_averagevar = mean(var(v)) #Compute average vote
maxvar[i]=maximum(var(v)) #Compute max vote variance 
averagevar[i]=compute_averagevar
var_array[i]=var(v) 
end 

averagevar_plot=plot(D_i_array, averagevar, xlabel="Diffusion Coefficient",
 ylabel="Mean Variance of Vote",
 lw=8, xlabelfontsize=20, ylabelfontsize=20, titlefontsize=12,
  legendfontsize=12, tickfontsize=16,
   legend=false)
display(averagevar_plot)
savefig("AverageVariance_DifferentD.pdf")
maxvar_plot=plot(D_i_array, maxvar, xlabel="Diffusion Coefficient",
 ylabel="Max Vote Variance", title="Max Variance of Vote vs Diffusion Coefficient",
 lw=8, xlabelfontsize=20, ylabelfontsize=20, titlefontsize=12,
  legendfontsize=12, tickfontsize=16,
   legend=false)
display(maxvar_plot)

log_plot=plot(D_i_array, log.(var_array), xlabel="Diffusion Coefficient",
 ylabel="Log Vote Variance",
 lw=8, xlabelfontsize=20, ylabelfontsize=20, titlefontsize=12,
  legendfontsize=12, tickfontsize=16,
   legend=false)
display(log_plot)


end #counter