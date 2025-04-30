using DifferentialEquations,NLsolve, Plots, Polynomials
# initial conditions
c0 = 1 #consensus makers party 1
g0 = 0 #gridlockers party 1
z0 = 0 #zealots party 1
ε0 = 0.25 #initial environmental state
v_c0 = 1  # initial v_c
v_g0 = 0  # initial v_g
u0 = [c0, g0, z0, ε0, v_c0, v_g0] 
# parameters
η = 0.2
r = 1.0
k = 1.0
p = (η, r, k) 
# time span
tspan = (0.0, 50.0)
# define the ODE problem
prob = ODEProblem(system!, u0, tspan, p)
function system!(du, u, p, t)
    # unpack variables
    c, g, z, ε, v_c, v_g = u
    η, r, k = p

    # Root finding way
    function equilibrium_eq(v, c, g, z)
        return v^2 * c + g * (1 - v)^2 + (z - v) * (2v^2 - 2v + 1)
    end
    roots_v = find_zeros(v -> equilibrium_eq(v, c, g, z), -0.1, 1.1)

    real_vs = [v for v in roots_v if isreal(v)]
    real_vs = real(roots_v)
    println(real_vs)

    if isempty(real_vs)
        # prevent division by zero if no real roots
        du .= 0.0
        return
    end

    # Set up
    c_dot = 0.0
    g_dot = 0.0
    z_dot = 0.0
    ε_dot = 0.0
    v_c_dot = 0.0
    v_g_dot = 0.0

    for v in real_vs
        # fitness functions
        f_c = ε * 4 * (v - 0.5)^2
        f_g = 1 - 4 * (v - 0.5)^2
        f_z  = v^2
        f_z2 = (1-ε)*(1-v)^2

        # Strategy Dynamics using replicator equation
        c_dot += c*g*(f_c - f_g) + c*z*(f_c - f_z) + c*(1 - c - g - z)*(f_c - f_z2)  
        g_dot += g*c*(f_g - f_c) + g*z*(f_g - f_z) + g*(1 - c - g - z)*(f_g - f_z2)  
        z_dot += z*c*(f_z - f_c) + z*g*(f_z - f_g)                                   
        ε_dot += r*ε*(1 - ε/k) - η*(1 - v)                                            

        # Function of environmental impact we should change this and see if it makes a difference
        f_ε = sin(ε)

        # Voting Dynamics
        v_c_dot += ε * ( (1 - v_c)*v*(v + f_ε) - v_c*(1 - v)*(2 - v - f_ε) )            
        v_g_dot += ε * ( (1 - v_g)*(1 - v)*(2 - v - f_ε) - v_g*v*(v + f_ε) )            
    end

    N = length(real_vs)
    du[1] = c_dot / N
    du[2] = g_dot / N
    du[3] = z_dot / N
    du[4] = ε_dot / N
    du[5] = v_c_dot / N
    du[6] = v_g_dot / N
end


# solve
sol = solve(prob, Tsit5())
# plot
plot(sol, xlabel="Time", ylabel="Proportions", label=["c" "g" "z" "ε" "v_c" "v_g"], lw=2, legend=:outertopright)