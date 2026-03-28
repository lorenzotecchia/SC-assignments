using SparseArrays
using LinearAlgebra
using Random
using Colors
using Images
using Plots
# using IterativeSolvers
# using Preconditioners 
# using IncompleteLU
using Statistics

# Suppress Qt warnings in WSL
ENV["QT_LOGGING_RULES"] = "*=false"
ENV["QT_QPA_PLATFORM"] = "xcb"

gr()  # Use GR backend explicitly for WSL compatibility

# speed of light
c = 3e8

# frequency (GHz)
f = 1.2e9

# wavelength
lambda = c / f

# grid resolution (meter per pixel)
#dx = 0.007
dx = lambda / 12

# material properties
air  = 1.0 + 0im
wall = 2.5 + 0.5im

# grid size (meters)
x_meter = 10.0
y_meter = 8.0

# grid size (pixels)
nx = Int(round(x_meter / dx))
ny = Int(round(y_meter / dx))

# wavenumber
k = 2*pi * f / c

# amplitude
A = 1e4

# width
sigma = 0.2

# measurement points
measurement_points = [(1., 5.,  "Living Room"),(2., 1., "Kitchen"), (9., 1., "Bathroom"), (9., 7., "Bedroom")]

# from meter to pixel
m2x(x) = Int(round(x/dx)) + 1
m2y(y) = Int(round(y/dx)) + 1

function create_floorplan()
    """
    creates the floor plan as the matrix floor, by converting meter into pixels using the predefined 
    grid resolution dx.
    - air = 1.0
    - wall = 0.0
    """ 

    floor = ones(Int, nx, ny)
    m2p_x(x) = clamp(Int(round(x / dx)) + 1, 1, nx)
    m2p_y(y) = clamp(Int(round(y / dx)) + 1, 1, ny)

    t_wall = 0.15
    t_pixel = max(1, Int(round(t_wall / dx)))

    floor[1:t_pixel, :] .= 0
    floor[end-t_pixel+1:end, :] .= 0
    floor[:, 1:t_pixel] .= 0
    floor[:, end-t_pixel+1:end] .= 0

    y = m2p_y(3.0)
    floor[:, y:y+t_pixel-1] .= 0

    floor[m2p_x(3.0):m2p_x(4.0), y:y+t_pixel-1] .= 1
    floor[m2p_x(6.0):m2p_x(7.0), y:y+t_pixel-1] .= 1

    x = m2p_x(2.5)
    floor[x:x+t_pixel-1, m2p_y(0.0):m2p_y(2.0)] .= 0

    x = m2p_x(6.0)
    floor[x:x+t_pixel-1, m2p_y(3.0):m2p_y(8.0)] .= 0

    x = m2p_x(7.0)
    floor[x:x+t_pixel-1, m2p_y(0.0):m2p_y(1.5)] .= 0
    floor[x:x+t_pixel-1, m2p_y(2.5):m2p_y(3.0)] .= 0

    return floor
end


function build_helmholtz_matrix(plan)
    """
    returns sparse Helmholtz matrix M for 2D domain 
        - Mu = f 
        - discretization: central finite difference scheme (5-point stencil)
        - uses Dirichlet boundary conditions (u=0)
    """

    # total number of grid points
    N = nx * ny

    # map 2D indices to 1D indices
    idx = LinearIndices((nx, ny))

    # sparse matrix (I=row, J=column, V=value)
    I = Int[]
    J = Int[]
    V = ComplexF64[]

    function add(i1,j1,i2, j2, val)
        # adds one matrix entry
        push!(I, idx[i1,j1])
        push!(J, idx[i2,j2])
        push!(V, val)
    end

    # discretization: finite difference
    # interior points
    for x in 1:nx, y in 1:ny
        # material type (air = 1, wall = 0)
        n = plan[x,y] == 1 ? air : wall

        # k^2 of helmholtz equation (n^2 is material properties)
        k2 = (k * n)^2

        if x == 1
            add(x, y, x, y, -1/dx + 1im*k)
            add(x, y, x+1, y, 1/dx)
            continue
        end

        if x == nx
            add(x, y, x, y, -1/dx - 1im*k)
            add(x, y, x-1, y, 1/dx)
            continue
        end

        if y == 1
            add(x, y, x, y, -1/dx + 1im*k)
            add(x, y, x, y+1, 1/dx)
            continue
        end
        
        if y == ny
            add(x, y, x, y, -1/dx - 1im*k)
            add(x, y, x, y-1, 1/dx)
            continue
        end

        # finite difference (5-point stencil)
        add(x,y,x,y, -4/dx^2 + k2)
        add(x,y,x+1,y, 1/dx^2)
        add(x,y,x-1,y, 1/dx^2)
        add(x,y,x,y+1, 1/dx^2)
        add(x,y,x,y-1, 1/dx^2)
    end

    return sparse(I, J, V, N, N)
end


function gaussian_source(rx, ry)
    """returns Wifi router source term f as Gaussian pulse."""

    f = zeros(ComplexF64, nx, ny)

    # router location
    router_x = m2x(rx)
    router_y = m2y(ry)

    # convert meters to grid units
    sigma_grid = sigma / dx

    # gaussian source
    for x in 1:nx, y in 1:ny
        # squared distance from router
        r2 = (x - router_x)^2 + (y - router_y)^2

        # source term (gaussian pulse)
        f[x,y] = A * exp(-r2 / (2*sigma_grid^2))
    end

    return vec(f)
end


function solve_Helmholtz_eq(rx, ry)
    """solves the Helmholtz equation and returns the wave field u."""

    floor = create_floorplan()
    M = build_helmholtz_matrix(floor)
    f = gaussian_source(rx, ry)

    # compute u = M^-1 * f and reshape to 2D
    u = reshape(M \ f, nx, ny)
    #P = ilu(M, τ=0.01)
    #u_vec, _ = gmres(M, f; Pl=P, restart=50, maxiter=500, tol=1e-6)

    return u, floor
end

function diagnose(u, floor)
    # to see if the signal is zero far from the router, ro just very low
    intensity = abs.(u).^2

    air_mask = floor .== 1
    i_air = intensity[air_mask]

    println("=== Intensity diagnostics (air only) ===")
    println("  max    : ", maximum(i_air))
    println("  min    : ", minimum(i_air))
    println("  median : ", median(i_air))
    println("  mean   : ", mean(i_air))
    println("  % below 1e-6 of max: ", 
        100 * mean(i_air .< 1e-6 * maximum(i_air)), "%")

    # also check the raw field (not intensity)
    println("\n=== Raw field u ===")
    println("  max |u| : ", maximum(abs.(u)))
    println("  min |u| : ", minimum(abs.(u[air_mask])))
end

function signal_strength(rx, ry)
    #=  Compute the total signal for router position (rx, ry). =#
    u, floor = solve_Helmholtz_eq(rx, ry)

    sum = 0.0
    for (mx, my, name) in measurement_points
        signal = measurement(u, mx, my)
        println("Signal in ", name, ": ", signal) 
        sum += signal
    end
    println("Total signal: ", sum) 
    return sum, u, floor
end

function measurement(u, mx, my)
    """ Performs one measurement in a 5cm radius region around the (mx, my) point."""
    mx = m2x(mx)
    my = m2y(my)

    radius = m2x(0.05)

    total = 0.
    for i in 0:radius
        for j in 0:radius
            if (i+j)^2<radius
                total += abs(u[mx+i, my+j])^2
            end
        end
    end
    return total
end

function check_position(rx, ry, floor)
    ix = m2x(rx)
    iy = m2y(ry)


    if ix <= 1 || iy <= 1
        return false 
    end
        if ix >= nx || iy >= ny
        return false 
    end
    if floor[m2x(rx), m2y(ry)] == 0.0
        return false
    end

    for (mx, my, _) in measurement_points
        if sqrt((mx-rx)^2 + (my-ry)^2) < 0.5    # radius is 0.5m
            return false
        end
    end
    return true
end

function get_neighbor(rx, ry, step_size, floor, tryals = 0)
    if tryals > 100
        return false, false
    end
    
    x_new = rx + rand()*2*step_size - step_size
    y_new = ry + rand()*2*step_size - step_size

    if check_position(x_new, y_new, floor)
        return x_new, y_new    
    else 
        return get_neighbor(x_new, y_new, step_size, floor, tryals + 1)
    end
end

function simulated_annealing(rx, ry, n_iterations, step_size, temp)
    best = [rx, ry]
    best_eval, best_u, floor = signal_strength(rx, ry)
    scores = [best_eval]

    current_eval = best_eval
    current = best
    for i in 1:n_iterations
        t = temp / (i) 
        candidate = get_neighbor(current[1], current[2], step_size, floor, 0)
        if candidate == (false, false)
            println("Annealing stopped since could not find optimal path.")
            return best, best_eval, scores
        end
        candidate_eval, candidate_u, _ = signal_strength(candidate[1], candidate[2])
        
        if candidate_eval > current_eval || rand() < exp((candidate_eval - current_eval) / t)
            current, current_eval = candidate, candidate_eval
            if candidate_eval > best_eval
                best, best_eval, best_u = candidate, candidate_eval, candidate_u
                push!(scores, best_eval)
            end
        end

        if i % 100 == 0
            println("iteration ", i, ", temperature: ", t:.3f, ", best evaluation: ", best_eval:.5f)
        end
    end
    return best, best_eval, best_u, scores, floor 
end

function show_heatmap(u, plan)

    # intensity = abs.(u).^2
    intensity = real.(u).^2

    # log scaling 
    intensity = log.(intensity .+ 1e-12)

    # mask walls
    intensity = Float64.(intensity)

    # white walls
    intensity[plan .== 0] .= NaN

    hmap = heatmap(
        intensity',
        color = :turbo,
    #    clim = (minimum(intensity), -20), 
        aspect_ratio = :equal,
        legend = :right,
        title = "WiFi Signal Strength (log intensity)",
        xlabel = "x",
        ylabel = "y"
    )

    for (x, y, _) in measurement_points
        scatter!([m2x(x)], [m2y(y)], markershape=:circle, markersize=6, color=:white, label="")
    end
    #savefig(hmap, "output/heatmap.png")
    hmap
    

end

function main()
    println("number of points per wavelength: ", round(lambda/dx))
    # router position
    rx = 7
    ry = 6

    n_iterations = 10
    step_size = 0.1 # in meters
    start_temperature = 1 

    best, best_eval, best_u, scores, floor = simulated_annealing(rx, ry, n_iterations, step_size, start_temperature)
     
    println("\n=== RESULT ===")
    println("best router position: ", best)
    println("best total signal strength: ", best_eval)

    # diagnose(u,floor)
    show_heatmap(best_u, floor)
end

main()