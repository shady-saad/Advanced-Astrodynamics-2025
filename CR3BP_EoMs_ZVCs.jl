# CR3BP_EoMs_ZVCs.jl
# Julia port of CR3BP_EoMs_ZVCs.m (Advanced Astrodynamics course)
# - Reimplements the MATLAB script in Julia
# - Requires: DifferentialEquations, Plots, LaTeXStrings
# Usage: julia CR3BP_EoMs_ZVCs.jl

# NOTE: This script attempts to install missing packages automatically.
import Pkg
for p in ["DifferentialEquations", "Plots", "LaTeXStrings"]
    try
        @eval using $(Symbol(p))
    catch err
        println("Package $p not found. Installing...")
        Pkg.add(p)
        @eval using $(Symbol(p))
    end
end

using LinearAlgebra
using DifferentialEquations
using Plots
using LaTeXStrings

# --------------------------- Utilities ---------------------------------
# Jacobi constant (matches MATLAB implementation)
function jacobi_constant(X::AbstractVector, mu::Real)
    x, y, z = X[1], X[2], X[3]
    xdot, ydot, zdot = X[4], X[5], X[6]
    r1 = sqrt((x + mu)^2 + y^2 + z^2)
    r2 = sqrt((x - 1 + mu)^2 + y^2 + z^2)
    C = (x^2 + y^2) + 2*(1-mu)/r1 + 2*mu/r2 - (xdot^2 + ydot^2 + zdot^2)
    return C
end

# CR3BP equations of motion in rotating frame
function cr3bp!(du, u, p, t)
    mu = p
    x, y, z = u[1], u[2], u[3]
    xdot, ydot, zdot = u[4], u[5], u[6]

    r1 = sqrt((x + mu)^2 + y^2 + z^2)
    r2 = sqrt((x - 1 + mu)^2 + y^2 + z^2)

    du[1] = xdot
    du[2] = ydot
    du[3] = zdot

    du[4] = 2*ydot + x - (1-mu)*(x + mu)/r1^3 - mu*(x + mu - 1)/r2^3
    du[5] = -2*xdot + y - (1-mu)*y/r1^3 - mu*y/r2^3
    du[6] = - (1-mu)*z/r1^3 - mu*z/r2^3
end

# Compute ZVC contour in xy-plane and return a contour object
function zvcxy_plot(mu::Real, C::Real; xrange=(-1.5,1.55), yrange=(-1.5,1.55), step=0.003)
    xs = collect(range(xrange[1], xrange[2], step=step))
    ys = collect(range(yrange[1], yrange[2], step=step))
    Z = Array{Float64}(undef, length(ys), length(xs))
    for (i,x) in enumerate(xs)
        for (j,y) in enumerate(ys)
            r1 = sqrt((x + mu)^2 + y^2)
            r2 = sqrt((x - 1 + mu)^2 + y^2)
            Z[j,i] = (x^2 + y^2) + 2*(1-mu)/r1 + 2*mu/r2
        end
    end
    # MATLAB code used contourf of -Z and level -C; here we'll plot contour of Z at value C.
    plt = contour(xs, ys, Z; levels=[C], linewidth=2, c=:black)
    return plt
end

# --------------------------- Problems ----------------------------------
function problem2()
    println("Running Problem 2 (trajectories + ZVC)...")
    # Initial conditions (normalized units as in MATLAB script)
    x0_1 = [0.98, 0.0, 0.0, 0.0, 1.2, 0.0]
    x0_2 = [0.98, 0.0, 0.0, 0.0, 1.7, 0.0]
    x0_3 = [0.12, 0.0, 0.0, 0.0, 3.45, 0.0]
    x0_4 = [0.12, 0.0, 0.0, 0.0, 3.48, 0.0]

    ToF_1 = 2.0
    ToF_2 = 8.0
    ToF_3 = 25.0
    ToF_4 = 25.0

    GM_earth = 398600.435436
    GM_moon = 4902.800066
    mu = GM_moon / (GM_earth + GM_moon)

    C1 = jacobi_constant(x0_1, mu)
    C2 = jacobi_constant(x0_2, mu)
    C3 = jacobi_constant(x0_3, mu)
    C4 = jacobi_constant(x0_4, mu)

    reltol = 2.22045e-14
    abstol = 2.22045e-16

    prob1 = ODEProblem(cr3bp!, x0_1, (0.0, ToF_1), mu)
    prob2 = ODEProblem(cr3bp!, x0_2, (0.0, ToF_2), mu)
    prob3 = ODEProblem(cr3bp!, x0_3, (0.0, ToF_3), mu)
    prob4 = ODEProblem(cr3bp!, x0_4, (0.0, ToF_4), mu)

    sol1 = solve(prob1, Vern9(); reltol=reltol, abstol=abstol)
    sol2 = solve(prob2, Vern9(); reltol=reltol, abstol=abstol)
    sol3 = solve(prob3, Vern9(); reltol=reltol, abstol=abstol)
    sol4 = solve(prob4, Vern9(); reltol=reltol, abstol=abstol)

    # Prepare subplots similar to MATLAB: 2x2
    plt = plot(layout=(2,2), size=(1000,800))

    p1 = plot(sol1[1,:], sol1[2,:], lw=1, c=:blue, label="Trajectory")
    zvc1 = zvcxy_plot(mu, C1; xrange=(0.97,1.005), yrange=(-0.05,0.05), step=0.0005)
    plot!(p1, zvc1)
    scatter!(p1, [1-mu], [0.0], color=:cyan, label="Moon", markersize=6)
    xlabel!(p1, L"x [-]"); ylabel!(p1, L"y [-]"); title!(p1, L"First x_0, C_J=$(round(C1, digits=5))")
    xlims!(p1, (0.97,1.005)); ylims!(p1, (-0.05,0.05))

    p2 = plot(sol2[1,:], sol2[2,:], lw=1, c=:red, label="Trajectory")
    zvc2 = zvcxy_plot(mu, C2; xrange=(0.8,1.2), yrange=(-0.2,0.2), step=0.001)
    plot!(p2, zvc2)
    scatter!(p2, [1-mu], [0.0], color=:cyan, label="Moon", markersize=6)
    xlabel!(p2, L"x [-]"); ylabel!(p2, L"y [-]"); title!(p2, L"Second x_0, C_J=$(round(C2, digits=5))")
    xlims!(p2, (0.8,1.2)); ylims!(p2, (-0.2,0.2))

    p3 = plot(sol3[1,:], sol3[2,:], lw=1, c=:green, label="Trajectory")
    zvc3 = zvcxy_plot(mu, C3)
    plot!(p3, zvc3)
    scatter!(p3, [-mu], [0.0], color=:yellow, label="Earth", markersize=6)
    scatter!(p3, [1-mu], [0.0], color=:cyan, label="Moon", markersize=6)
    xlabel!(p3, L"x [-]"); ylabel!(p3, L"y [-]"); title!(p3, L"Third x_0, C_J=$(round(C3, digits=5))")

    p4 = plot(sol4[1,:], sol4[2,:], lw=1, c=:magenta, label="Trajectory")
    zvc4 = zvcxy_plot(mu, C4)
    plot!(p4, zvc4)
    scatter!(p4, [-mu], [0.0], color=:yellow, label="Earth", markersize=6)
    scatter!(p4, [1-mu], [0.0], color=:cyan, label="Moon", markersize=6)
    xlabel!(p4, L"x [-]"); ylabel!(p4, L"y [-]"); title!(p4, L"Fourth x_0, C_J=$(round(C4, digits=5))")

    plot!(plt, p1, subplot=1)
    plot!(plt, p2, subplot=2)
    plot!(plt, p3, subplot=3)
    plot!(plt, p4, subplot=4)

    savefig(plt, "Prob2.pdf")
    println("Saved Prob2.pdf")
end

function problem2c()
    println("Running Problem 2C (Jacobi constant conservation analysis)...")
    # same initials as problem 2
    x0_1 = [0.98, 0.0, 0.0, 0.0, 1.2, 0.0]
    x0_2 = [0.98, 0.0, 0.0, 0.0, 1.7, 0.0]
    x0_3 = [0.12, 0.0, 0.0, 0.0, 3.45, 0.0]
    x0_4 = [0.12, 0.0, 0.0, 0.0, 3.48, 0.0]

    ToF_1 = 2.0; ToF_2 = 8.0; ToF_3 = 25.0; ToF_4 = 25.0

    GM_earth = 398600.435436
    GM_moon = 4902.800066
    mu = GM_moon / (GM_earth + GM_moon)

    len = 10000
    t1 = range(0, ToF_1, length=len)
    t2 = range(0, ToF_2, length=len)
    t3 = range(0, ToF_3, length=len)
    t4 = range(0, ToF_4, length=len)

    prob1 = ODEProblem(cr3bp!, x0_1, (0.0, ToF_1), mu)
    prob2 = ODEProblem(cr3bp!, x0_2, (0.0, ToF_2), mu)
    prob3 = ODEProblem(cr3bp!, x0_3, (0.0, ToF_3), mu)
    prob4 = ODEProblem(cr3bp!, x0_4, (0.0, ToF_4), mu)

    sol1 = solve(prob1, Vern9(), saveat=t1)
    sol2 = solve(prob2, Vern9(), saveat=t2)
    sol3 = solve(prob3, Vern9(), saveat=t3)
    sol4 = solve(prob4, Vern9(), saveat=t4)

    C1_vec = [jacobi_constant(sol1.u[i], mu) for i in 1:length(sol1.u)]
    C2_vec = [jacobi_constant(sol2.u[i], mu) for i in 1:length(sol2.u)]
    C3_vec = [jacobi_constant(sol3.u[i], mu) for i in 1:length(sol3.u)]
    C4_vec = [jacobi_constant(sol4.u[i], mu) for i in 1:length(sol4.u)]

    cumRelStd_C1 = [std(C1_vec[1:i]) / abs(mean(C1_vec[1:i])) for i in 1:length(C1_vec)]
    cumRelStd_C2 = [std(C2_vec[1:i]) / abs(mean(C2_vec[1:i])) for i in 1:length(C2_vec)]
    cumRelStd_C3 = [std(C3_vec[1:i]) / abs(mean(C3_vec[1:i])) for i in 1:length(C3_vec)]
    cumRelStd_C4 = [std(C4_vec[1:i]) / abs(mean(C4_vec[1:i])) for i in 1:length(C4_vec)]

    samples = 1:length(cumRelStd_C1)
    plt = plot(size=(900,500))
    plot!(samples, cumRelStd_C1, lw=1, c=:magenta, label="First x0, ToF=2")
    plot!(samples, cumRelStd_C2, lw=1, c=:red, label="Second x0, ToF=8")
    plot!(samples, cumRelStd_C3, lw=1, c=:blue, label="Third x0, ToF=25")
    plot!(samples, cumRelStd_C4, lw=1, c=:green, label="Fourth x0, ToF=25")
    yaxis!(:log10)
    xlabel!(L"Samples [-]"); ylabel!(L"σ_C/μ_C [-]")
    legend(loc=:lowerright)
    savefig(plt, "Prob2C.pdf")
    println("Saved Prob2C.pdf")
end

function problem3()
    println("Running Problem 3 (event detection)...")
    x0_3 = [0.12, 0.0, 0.0, 0.0, 3.45, 0.0]
    ToF_3 = 25.0
    GM_earth = 398600.435436
    GM_moon = 4902.800066
    mu = GM_moon / (GM_earth + GM_moon)

    # Event: y = 0 with positive crossing (ydot>0). Use ContinuousCallback with direction=1
    condition(u,t,integ) = u[2] # y
    affect!(integ) = terminate!(integ)
    cb = ContinuousCallback(condition, affect!; direction=1)

    prob = ODEProblem(cr3bp!, x0_3, (0.0, ToF_3), mu)
    sol = solve(prob, Vern9(); callback=cb, reltol=2.22045e-14, abstol=2.22045e-16)

    # Plot trajectory and ZVC
    C3 = jacobi_constant(x0_3, mu)
    plt = plot(sol[1,:], sol[2,:], lw=1, c=:green, label="Trajectory")
    zvc3 = zvcxy_plot(mu, C3)
    plot!(plt, zvc3)
    scatter!([-mu, 1-mu], [0.0,0.0], color=[:yellow,:cyan], label=["Earth","Moon"])
    xlabel!(L"x [-]"); ylabel!(L"y [-]")
    title!(L"Third x_0 with event: y=0, ydot>0")
    savefig(plt, "Prob3.pdf")
    println("Saved Prob3.pdf")

    # Results at event
    if sol.t[end] < ToF_3
        println("Event detected at t = ", sol.t[end])
        println("State at event: ", sol.u[end])
    else
        println("No event in time of flight.")
    end
end

function problem4()
    println("Running Problem 4 (ZVCs for given Jacobi constants)...")
    GM_earth = 398600.435436
    GM_moon = 4902.800066
    mu = GM_moon / (GM_earth + GM_moon)

    C1 = 3.189
    C2 = 3.173
    C3 = 3.013
    C4 = 2.995

    plt = plot(layout=(2,2), size=(900,700))
    plot!(plt[1], zvcxy_plot(mu,C1), title=L"C_J = 3.189")
    scatter!(plt[1], [-mu, 1-mu], [0.0,0.0], color=[:blue,:red], ms=6, label=["Earth","Moon"])    
    plot!(plt[2], zvcxy_plot(mu,C2), title=L"C_J = 3.173")
    scatter!(plt[2], [-mu, 1-mu], [0.0,0.0], color=[:blue,:red], ms=6, label=["Earth","Moon"])    
    plot!(plt[3], zvcxy_plot(mu,C3), title=L"C_J = 3.013")
    scatter!(plt[3], [-mu, 1-mu], [0.0,0.0], color=[:blue,:red], ms=6, label=["Earth","Moon"])    
    plot!(plt[4], zvcxy_plot(mu,C4), title=L"C_J = 2.995")
    scatter!(plt[4], [-mu, 1-mu], [0.0,0.0], color=[:blue,:red], ms=6, label=["Earth","Moon"])    

    savefig(plt, "Prob4.pdf")
    println("Saved Prob4.pdf")
end

function problem4d()
    println("Running Problem 4D (Lagrange point zoom and manual picking)...")
    GM_earth = 398600.435436
    GM_moon = 4902.800066
    mu = GM_moon / (GM_earth + GM_moon)

    C1 = 3.18841
    C2 = 3.1722
    C3 = 3.0124
    C45 = 2.988

    plt = plot(layout=(2,2), size=(900,700))
    p1 = zvcxy_plot(mu, C1; xrange=(0.832,0.842), yrange=(-0.02,0.02), step=0.00025)
    plot!(plt[1], p1); title!(plt[1], L"L1 Position")
    p2 = zvcxy_plot(mu, C2; xrange=(1.151,1.161), yrange=(-0.02,0.02), step=0.00025)
    plot!(plt[2], p2); title!(plt[2], L"L2 Position")
    p3 = zvcxy_plot(mu, C3; xrange=(-1.05,-0.95), yrange=(-0.01,0.01), step=0.00025)
    plot!(plt[3], p3); title!(plt[3], L"L3 Position")
    p4 = zvcxy_plot(mu, C45; xrange=(0.44,0.54), yrange=(0.861,0.871), step=0.00025)
    plot!(plt[4], p4); title!(plt[4], L"L4 Position")

    savefig(plt, "Prob4D.pdf")
    println("Saved Prob4D.pdf")
    println("NOTE: MATLAB used ginput to manually pick points. In this Julia script we don't provide interactive picking. If you want to pick, use GUI display and interact with the Figures.")
end

# --------------------------- Main --------------------------------------
function main()
    println("CR3BP Julia port -- starting")
    # Run problems sequentially. Users can comment out calls they don't want to run.
    problem2()
    problem2c()
    problem3()
    problem4()
    problem4d()
    println("Done. Generated PDF files: Prob2.pdf, Prob2C.pdf, Prob3.pdf, Prob4.pdf, Prob4D.pdf (in current directory).")
end

# run main when script executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
