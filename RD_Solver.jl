using DifferentialEquations, Plots, LinearAlgebra, Roots
@time begin
# Parameters
D = 0.001 #Diffusion Coefficient
L = 1.0 #Length of the domain
ξ=100000 #Big number for fast convergence
Nx, Ny = 10, 10 #Numer of discretization points in either direction
dx = L / (Nx - 1) #Chop up x
dy = L / (Ny - 1) #Chop up y
x = range(0, L, length=Nx)
y = range(0, L, length=Ny)
X, Y = [xi for xi in x, yi in y], [yi for xi in x, yi in y]

# Gaussian shell
function gaussian(x, y, x0, y0, sigma)
    return exp.(-((x .- x0).^2 + (y .- y0).^2) ./ (2*sigma^2))
end
#Initial distribution/ conditions
var = 0.1 #variance
N=Nx
c₀ = rand(N,N) #gaussian(X, Y, 0.5, 0.5, var) #initial distribution for Consensus makers
g₀ = rand(N,N) #gaussian(X, Y, 0.5, 0.4, var) #initial distribution for Gridlockers
z1₀ = rand(N,N) #gaussian(X, Y, 0.5, 0.5, var) #initial distribution for Zealots Party 1
z2₀ = rand(N,N) #gaussian(X, Y, 0.5, 0.5, var) #initial distribution for Zealots Party 2
τ=c₀+g₀+z1₀
c₀=c₀ ./ τ
g₀=g₀ ./ τ
z1₀=z1₀ ./ τ
v_c₀=rand(N,N)
v_g₀=rand(N,N)
function pack(c, g, z,v_c,v_g)
    return vcat(vec(c), vec(g), vec(z),vec(v_c),vec(v_g))
end

function unpack(u)
    N = Nx * Ny
    c     = reshape(u[1:N], Nx, Ny)
    g     = reshape(u[N+1:2N], Nx, Ny)
    z     = reshape(u[2N+1:3N], Nx, Ny)
    v_g   = reshape(u[3N+1:4N], Nx, Ny)
    v_c   = reshape(u[4N+1:5N], Nx, Ny)
    return c, g, z, v_g, v_c
end

#Construct laplacian in 2D
function laplacian(U)
    L = similar(U)
    for i in 2:Nx-1, j in 2:Ny-1
        L[i,j] = (U[i+1,j] + U[i-1,j] + U[i,j+1] + U[i,j-1] - 4U[i,j]) / dx^2
    end
    L[1,:] .= L[2,:]; L[end,:] .= L[end-1,:]
    L[:,1] .= L[:,2]; L[:,end] .= L[:,end-1]
    return L
end

# Define scalar fitness functions
Fitness_c(v) = 4 * (v .- 0.5).^2 #Strategy fitness for Consensus makers
Fitness_g(v) = 1 .- 4 * (v .- 0.5).^2 #Strategy fitness for Gridlockers
Fitness_z1(v) = (v).^2 #Strategy fitness for Zealots party 1
Fitness_z2(v) = 1-(v).^2 #Strategy fitness for Zealots party 2
#Set up equilibrium equation
function equilibrium_eq(v, c, g, z)
    return v^2 * c + g * (1 - v)^2 + (z - v) * (2v^2 - 2v + 1)
end

# Compute the equilibrium at each discretization point
# v_real = zeros(Nx, Ny)
# for i in 1:Nx, j in 1:Ny
#     local_c, local_g, local_z = c₀[i,j], g₀[i,j], z₀[i,j] #set up local computations
#     v_real[i,j] = find_zero(v -> equilibrium_eq(v, local_c, local_g, local_z), 0.5) #roots package used
# end

# Initial condition
u0 = pack(c₀, g₀, z1₀,v_c₀,v_g₀)
tspan = (0.0, 0.1)

# Define the PDE system using v_real as a fixed spatial array
function pdes!(du, u, p, t)
    c, g, z,v_c,v_g = unpack(u)
    # Fitness values are matrices now
    v=c.*v_c+g.*v_g+z
    F_c = Fitness_c(v)
    F_g = Fitness_g(v)
    F_z = Fitness_z1(v)
    # Reaction Diffusion equation with Replicator Equation
    du_c = D * laplacian(c) + (c .* g .* (F_c .- F_g) + c .* z .* (F_c .- F_z))
    du_g = D * laplacian(g) + (g .* c .* (F_g .- F_c) + g .* z .* (F_g .- F_z))
    du_z = D * laplacian(z) + (z .* c .* (F_z .- F_c) + z .* g .* (F_z .- F_g))
    v_c = ξ* ((1 .-v_c).*v^2+v_c.*(1 .-v)^2)
    v_g = ξ* (-(v_g).*v^2+(1 .-v_g).*(1 .-v)^2)
    du .= pack(du_c, du_g, du_z,v_c,v_g)
end

#Solve the system of PDE's
prob = ODEProblem(pdes!, u0, tspan)
sol = solve(prob, Tsit5(), saveat=0.01)

#Plot results
c, g, z = unpack(sol[end])
p1 = heatmap(x, y, c', title="c(x,y)", xlabel="x", ylabel="y", aspect_ratio=1)
p2 = heatmap(x, y, g', title="g(x,y)", xlabel="x", ylabel="y", aspect_ratio=1)
p3 = heatmap(x, y, z', title="z(x,y)", xlabel="x", ylabel="y", aspect_ratio=1)
plot(p1, p2, p3, layout=(1,3), size=(1200, 400), plot_title="Solutions at final time 0.01 , with D=$D and var =$var ")
end
