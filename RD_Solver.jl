using DifferentialEquations, Plots, LinearAlgebra, Roots, Statistics, Sundials, ColorSchemes
@time begin
# Parameters for computations
D_c = 1e-3#3 #Diffusion Coefficient for Consensus makers
D_g = 1e-3#3 #Diffusion Coefficient for Gridlockers
D_z = 1e-3#3 #Diffusion Coefficient for Zealots
D_z2 = 1e-3#3 #Diffusion Coefficient for Zealots Party 2
m_c = 0#1e-3 #Migration rate for Consensus makers
m_g = 0#1e-3 #Migration rate for Gridlockers
m_z = 0#1e-3 #Migration rate for Zealots Party 1
m_z2 =0# 1e-3 #Migration rate for Zealots Party 2
V=1 #Social Imitation
λ= 0 #0.5 #Economic preference
b=1#public good benefit
k=0 #public good cost
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
#c₀ = clamp.(c₀, 0, .25) #Control the bounds of initial conditions
g₀ = rand(N,N) #gaussian(X, Y, 0.5, 0.4, var) #initial distribution for Gridlockers
#g₀ = clamp.(g₀, 0, 0.05) #Control the bounds of initial conditions, we can change these around to see what will happen
z1₀ = rand(N,N) #gaussian(X, Y, 0.5, 0.5, var) #initial distribution for Zealots Party 1
#z1₀ = clamp.(z1₀, 0, 0.25) #Control the bounds of initial conditions, we can change these around to see what will happen
z2₀ = rand(N,N) #gaussian(X, Y, 0.5, 0.5, var) #initial distribution for Zealots Party 2
#z2₀ = clamp.(z2₀, 0, 0.05) #Control the bounds of initial conditions, we can change these around to see what will happen
τ= c₀ .+ g₀ .+ z1₀ .+ z2₀
c₀= round.(rand(N,N))./ τ
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

# Fitness functions
Fitness_c(v) = 4 .* (v .- 0.5).^2 #Strategy fitness for Consensus makers
Fitness_g(v) = 1 .- 4 .* (v .- 0.5).^2 #Strategy fitness for Gridlockers
Fitness_z1(v) = (v).^2 #Strategy fitness for Zealots party 1
Fitness_z2(v) = (1 .- (v)).^2 #Strategy fitness for Zealots party 2

#Economic Utility functions
Utility_c(v) = λ .*(b-k) .* v+(1-λ).*Fitness_c(v) #Utility for Consensus makers
Utility_g(v) = λ .*(b-k) .* v+(1-λ).*Fitness_g(v) #Utility for Gridlockers
Utility_z1(v) = λ .*(b-k) .* v+(1-λ).*Fitness_z1(v) #Utility for Zealots party 1
Utility_z2(v) = λ .*(b-k) .* v+(1-λ).*Fitness_z2(v) #Utility for Zealots party 2
# Initial condition
u0 = pack(c₀, g₀, z1₀, z2₀, v_c₀, v_g₀)
du0 = zeros(size(u0))
time = (0.0, tfinal)

end

function DAE!(du, u, p, t)
    c, g, z, z2, v_c, v_g = unpack(u)
    # du_c, du_g, du_z, du_v_c, du_v_g = unpack(du)
    v = c .* v_c .+ g .* v_g .+ z #vote for party 1

    F_c = Fitness_c(v)
    F_g = Fitness_g(v)
    F_z = Fitness_z1(v)
    F_z2 = Fitness_z2(v)
    u_c = Utility_c(v)
    u_g = Utility_g(v)
    u_z1 = Utility_z1(v)
    u_z2 = Utility_z2(v)
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
    # Partial Differential equations
    du_c = D_c .* laplacian(c) .-m_c .*div_c_grad_uc .+ V.* (c .* g .* (F_c .- F_g) + c .* z .* (F_c .- F_z) + c .* z2 .* (F_c .- F_z2))
    du_g = D_g .* laplacian(g) .-m_g .*div_g_grad_ug .+ V.* (g .* c .* (F_g .- F_c) + g .* z .* (F_g .- F_z) + g .* z2 .* (F_g .- F_z2))
    du_z = D_z .* laplacian(z) .-m_z .*div_z1_grad_uz1 .+ V.* (z .* c .* (F_z .- F_c) + z .* g .* (F_z .- F_g))
    du_z2 = D_z2 .* laplacian(z2) .-m_z2 .*div_z2_grad_uz2 .+ V.* (z2 .* c .* (F_z2 .- F_c) + z2 .* g .* (F_z2 .- F_g))
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
M = diagm(vcat(ones(6*Nx*Ny)))

u0 = pack(c₀, g₀, z1₀, z2₀, v_c₀, v_g₀)
du0 = zeros(size(u0))
time = (0.0, tfinal)

DAEfunc = ODEFunction(DAE!, mass_matrix = M)
prob = ODEProblem(DAEfunc, u0, time)
sol = solve(prob, RadauIIA5(), saveat=0.01, reltol=1e-6, abstol=1e-6) #Different solvers:RadauIIA5,Rodas5P,Rodas4,ROS34PW2,ROS34PW3,Trapezoid

# HEATMAPS: Plot results at final time 
# percent_change_heatmap_population = []
# for idx in 1:length(sol)
#     c, g, z, z2, v_c, v_g = unpack(sol[idx])
#     population = c .+ g .+ z .+ z2
#     current_heatmap_population = population ./ total_population
#     percent_change = 100 .* (current_heatmap_population .-τ) ./ τ
#     push!(percent_change_heatmap_population, percent_change)
# end

fontsize=14
c, g, z, z2, v_c, v_g = unpack(sol[end]) #computations from the end of the simulation, we could pull these at any other times
population = c .+ g .+ z .+ z2 #Compute population, this is a matrix
total_population = sum(c) .+ sum(g) .+ sum(z) .+ sum(z2) #Compute population at the end, this is a scalar
c=c ./ population #Normalize c
g=g ./ population #Normalize g
z=z ./ population #Normalize z
z2=z2 ./ population #Normalize z2
heatmap_population = population ./ total_population #Normalize population
v = (c .* v_c .+ g .* v_g .+ z) #Compute v at the end
clims = (0, 1) #Color limits for heatmaps
p1 = heatmap(x, y, c',  aspect_ratio=1,colorbar=false, clims=clims)# clims=clims
p2 = heatmap(x, y, g',  aspect_ratio=1,colorbar=false, clims=clims)# clims=clims
p3 = heatmap(x, y, z',  aspect_ratio=1,colorbar=false, clims=clims)# clims=clims
p4 = heatmap(x, y, z2',  aspect_ratio=1,colorbar=false, clims=clims)# clims=clims
p5 = heatmap(x, y, heatmap_population',  aspect_ratio=1,colorbar=false, clims=clims) # clims=clims
p6 = heatmap(x, y, v',  aspect_ratio=1,color=:cool, colorbar=false, clims=clims) # clims=climscolor=:balance,
heatmap_figure = plot(p1, p2, p3, p4, p5, p6, layout=(3,3), size=(1400, 1500),colorbar=true, titlefontsize=fontsize, guidefontsize=fontsize, tickfontsize=fontsize, plot_title="Solutions at final time $tfinal")
display(plot(p1, axis=false, framestyle=:none,ticks=false, size=(625, 625))) #Consensus makers
#savefig("Heatmap4DifferentM_D=0_C,lambda=$λ,T=$tfinal.pdf")
display(plot(p2, axis=false, framestyle=:none, ticks=false,size=(625, 625))) #Gridlockers
#savefig("Heatmap4DifferentM_D=0_G,lambda=$λ,T=$tfinal.pdf")
display(plot(p3, axis=false, framestyle=:none, ticks=false,size=(625, 625))) #Zealots of party 1
#savefig("Heatmap4DifferentM_D=0_Z1,lambda=$λ,T=$tfinal.pdf");
display(plot(p4, axis=false, framestyle=:none, ticks=false,size=(625, 625))) #Zealots of party 2
#savefig("Heatmap4DifferentM_D=0_Z2,lambda=$λ,T=$tfinal.pdf")
display(plot(p5, axis=false, framestyle=:none, ticks=false,size=(625, 625))) #Population
#savefig("Heatmap4DifferentM_D=0_Population,lambda=$λ,T=$tfinal.pdf")
display(plot(p6, axis=false, framestyle=:none, ticks=false, size=(625,625))) #Vote
#savefig("Heatmap4DifferentM_D=0_Vote,lambda=$λ,T=$tfinal.pdf")
display(heatmap_figure)
#savefig("Heatmap_Clean_DifferentD_EvenIC_Finaltime=$tfinal.pdf")

# TIME SERIES: Compute averages over the domain at each time step
time_steps = sol.t
average_c = [mean(unpack(sol[i])[1]) for i in 1:length(time_steps)] #Average Consensus-makers
average_g = [mean(unpack(sol[i])[2]) for i in 1:length(time_steps)] #Average Gridlockers
average_z = [mean(unpack(sol[i])[3]) for i in 1:length(time_steps)] #Average Zealots of party 1
average_z2 = [mean(unpack(sol[i])[4]) for i in 1:length(time_steps)] #Average Zealots of party 2
average_Fitness_z1 = [mean(Fitness_z1(unpack(sol[i])[5])) for i in 1:length(time_steps)]
average_Fitness_z2 = [mean(Fitness_z2(unpack(sol[i])[5])) for i in 1:length(time_steps)]
average_Fitness_c = [mean(Fitness_c(unpack(sol[i])[5])) for i in 1:length(time_steps)]
average_Fitness_g = [mean(Fitness_g(unpack(sol[i])[5])) for i in 1:length(time_steps)]
#ts_max_pop = [maximum(unpack(sol[i])[1]) + maximum(unpack(sol[i])[2]) + maximum(unpack(sol[i])[3]) + maximum(unpack(sol[i])[4]) for i in 1:length(time_steps)]
average_v = [mean(unpack(sol[i])[5]) .* mean(unpack(sol[i])[1])  .+ mean(unpack(sol[i])[2]) .* mean(unpack(sol[i])[6]) .+ mean(unpack(sol[i])[3]) .+ mean(unpack(sol[i])[4]) for i in 1:length(time_steps)]
# Above computes c*v_c + g*v_g + z at each time step
# Plot averages
time_series = plot(time_steps, average_c, xlabel="Time", ylabel="Mean",lw=8, xlabelfontsize=20, ylabelfontsize=20,
     titlefontsize=12, legendfontsize=12, tickfontsize=16, legend=false) #, label="Mean Consensus Makers"
plot!(time_steps, average_g,lw=8)
plot!(time_steps, average_z,lw=8)
plot!(time_steps, average_z2,lw=8)
plot!(time_steps, average_v,lw=8)
# plot!(time_steps, average_Fitness_z1,lw=8)
# plot!(time_steps, average_Fitness_z2,lw=8)
#plot!(time_steps, ts_max_pop, label="Max Population",lw=3)
display(time_series)
#savefig("TimeSeriesGridlockT=$tfinal.pdf")

## Time series fitness
time_series_fit = plot(time_steps, average_Fitness_c, xlabel="Time", ylabel="Mean",lw=8, xlabelfontsize=20, ylabelfontsize=20,
     titlefontsize=12, legendfontsize=12, tickfontsize=16, label="Mean Consensus Makers Fitness")
plot!(time_steps, average_Fitness_g,label="Mean Gridlockers Fitness",lw=8)
plot!(time_steps, average_Fitness_z1,label="Mean Zealots 1 Fitness",lw=8)
plot!(time_steps, average_Fitness_z2,label="Mean Zealots 2 Fitness",lw=8)
#plot!(time_steps, ts_max_pop, label="Max Population",lw=3)
display(time_series_fit)



#Sanity check: Plot the average v_c and v_g 
mean_vc = [mean(unpack(sol[i])[5]) for i in 1:length(time_steps)]
mean_vg = [mean(unpack(sol[i])[6]) for i in 1:length(time_steps)]
min_vc = [minimum(unpack(sol[i])[5]) for i in 1:length(time_steps)]
min_vg = [minimum(unpack(sol[i])[6]) for i in 1:length(time_steps)]
max_vc = [maximum(unpack(sol[i])[5]) for i in 1:length(time_steps)]
max_vg = [maximum(unpack(sol[i])[6]) for i in 1:length(time_steps)]
sanity_population = [sum(unpack(sol[i])[1]) + sum(unpack(sol[i])[2]) + sum(unpack(sol[i])[3]) + sum(unpack(sol[i])[4]) for i in 1:length(time_steps)] ./ [sum(unpack(sol[1])[1]) + sum(unpack(sol[1])[2]) + sum(unpack(sol[1])[3]) + sum(unpack(sol[1])[4])] #./[sum(unpack(sol[1])[1]) + sum(unpack(sol[1])[2]) + sum(unpack(sol[1])[3]) + sum(unpack(sol[1])[4])] # Normalize population
Population_error = maximum(sanity_population) - minimum(sanity_population)
SanityCheck=plot(time_steps, mean_vc, label="Mean v_c", xlabel="Time", ylabel="Mean", lw=3, legend=:outertopright, title="Mean v_c and v_g over time, Sanity Check")
plot!(time_steps, mean_vg, label="Mean v_g", lw=3)
plot!(time_steps, min_vc, label="Min v_c", lw=3)
plot!(time_steps, min_vg, label="Min v_g", lw=3)
plot!(time_steps, max_vc, label="Max v_c", lw=3)
plot!(time_steps, max_vg, label="Max v_g", lw=3)
plot!(time_steps, sanity_population, label="Population", lw=3)
#display(SanityCheck)
#savefig("MeanTS(SanityCheck)_D_0.01_Finaltime=$tfinal.pdf")

#println("Population error: ", Population_error)


nframes = 50
frame_idxs = round.(Int, range(1, length(sol), length=nframes))

anim = @animate for idx in frame_idxs
    c, g, z, z2, v_c, v_g = unpack(sol[idx])
    population = c .+ g .+ z .+ z2
    c = c ./ population
    g = g ./ population
    z = z ./ population
    z2 = z2 ./ population
    heatmap_population = population ./ sum(population)
    v = (c .* v_c .+ g .* v_g .+ z)
    clims = (0, 1)
    p1 = heatmap(x, y, c', aspect_ratio=1, colorbar=false, clims=clims)
    p2 = heatmap(x, y, g', aspect_ratio=1, colorbar=false, clims=clims)
    p3 = heatmap(x, y, z', aspect_ratio=1, colorbar=false, clims=clims)
    p4 = heatmap(x, y, z2', aspect_ratio=1, colorbar=false, clims=clims)
    p5 = heatmap(x, y, heatmap_population', aspect_ratio=1, colorbar=false, clims=clims)
    p6 = heatmap(x, y, v', aspect_ratio=1, colorbar=false, color=:cool, clims=clims)
    plot(p1, p2, p3, p4, p5, p6, layout=(3,3), size=(1400, 1500), colorbar=true, titlefontsize=fontsize, guidefontsize=fontsize, tickfontsize=fontsize, plot_title="Solutions at t=$(round(sol.t[idx], digits=2))")
end

mp4(anim, "heatmap_video.mp4", fps=5)