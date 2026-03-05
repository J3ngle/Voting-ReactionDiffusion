#VariancePlots
using DifferentialEquations, Plots, LinearAlgebra, Roots, Statistics, Sundials, ColorSchemes, SparseArrays
@time begin

    iter = 5
    D_i_array = range(1e-3, 0.1, length=iter) # Diffusion coefficient for all types
    P = 1 # Number of iterations per D_i
    averagevar = zeros(iter)
    maxvar = zeros(iter)
    var_array = zeros(iter)
    averagevar_per_D = zeros(iter)
    for i in 1:iter
        D_i = D_i_array[i]
        averagevar_P = zeros(P)
        for p in 1:P
            local D_c, D_g, D_z, D_z2, m_c, m_g, m_z, m_z2, V, λ, b, k, s, L, Nx, Ny, dx, dy, x, y, tfinal, X, Y, N, c₀, g₀, z1₀, z2₀, τ, v_0, v_c₀, v_g₀, u0, du0, tspan, sol, c, g, z, z2, v_c, v_g, population, total_population, heatmap_population, v
            D_c = D_i
            D_g = D_i
            D_z = D_i
            D_z2 = D_i
            m_c = 0
            m_g = 0
            m_z = 0
            m_z2 = 0
            V = 1
            λ = 0
            b = 0
            k = 0
            s = 0
            L = 10
            Nx, Ny = 10, 10
            dx = L / (Nx - 1)
            dy = L / (Ny - 1)
            x = range(0, L, length=Nx)
            y = range(0, L, length=Ny)
            tfinal = 25.0
            X, Y = [xi for xi in x, yi in y], [yi for xi in x, yi in y]
            N = Nx
            c₀ = rand(N, N)
            g₀ = rand(N, N)
            z1₀ = rand(N, N)
            z2₀ = rand(N, N)
            τ = c₀ .+ g₀ .+ z1₀ .+ z2₀
            c₀ = c₀ ./ τ
            g₀ = g₀ ./ τ
            z1₀ = z1₀ ./ τ
            z2₀ = z2₀ ./ τ
            v_0 = c₀ .+ 0.5 * g₀ .+ z1₀
            v_c₀ = v_0.^2 ./(2 .* v_0.^2 .- 2 .* v_0 .+ 1)
            v_g₀ = (1 .- v_0).^2 ./ (2 .* v_0.^2 .- 2 .* v_0 .+ 1)
            u0 = pack(c₀, g₀, z1₀, z2₀, v_c₀, v_g₀)
            du0 = zeros(size(u0))
            tspan = (0.0, tfinal)
            DAEfunc = ODEFunction(DAE!, mass_matrix = M)
            prob = ODEProblem(DAEfunc, u0, tspan)
            sol = solve(prob, RadauIIA5(), saveat=0.01, reltol=1e-12, abstol=1e-12)
            c, g, z, z2, v_c, v_g = unpack(sol[end], Nx, Ny)
            population = c .+ g .+ z .+ z2
            total_population = sum(c) .+ sum(g) .+ sum(z) .+ sum(z2)
            c = c ./ population
            g = g ./ population
            z = z ./ population
            z2 = z2 ./ population
            v = (c .* v_c .+ g .* v_g .+ z)
            averagevar_P[p] = mean(var(v))
        end
        averagevar_per_D[i] = mean(averagevar_P)
    end
    # Plotting
    averagevar_plot = plot(D_i_array, averagevar_per_D, xlabel="Diffusion Coefficient",
        ylabel="Mean Variance of Vote (Averaged)",
        lw=8, xlabelfontsize=20, ylabelfontsize=20, titlefontsize=12,
        legendfontsize=12, tickfontsize=16,
        legend=false)
    display(averagevar_plot)
    savefig("AverageVariance_DifferentD_P.pdf")
    # ...existing code...
function pack(c, g, z, z2, v_c, v_g)
    # Assumes all matrices are the same size
    Nx, Ny = size(c)
    return vcat(vec(c), vec(g), vec(z), vec(z2), vec(v_c), vec(v_g))
end

function unpack(u, Nx, Ny)
    N = Nx * Ny
    c   = reshape(u[1:N], Nx, Ny)
    g   = reshape(u[N+1:2N], Nx, Ny)
    z   = reshape(u[2N+1:3N], Nx, Ny)
    z2  = reshape(u[3N+1:4N], Nx, Ny)
    v_c = reshape(u[4N+1:5N], Nx, Ny)
    v_g = reshape(u[5N+1:6N], Nx, Ny)
    return c, g, z, z2, v_c, v_g
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
        c, g, z, z2, v_c, v_g = unpack(sol[i], Nx, Ny)
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
    c, g, z, z2, v_c, v_g = unpack(u, Nx, Ny)
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