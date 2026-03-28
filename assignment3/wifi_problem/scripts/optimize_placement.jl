"""
Simulated annealing optimization for WiFi router placement.

This script uses simulated annealing to find the optimal router position
that maximizes total signal strength at all measurement points.
"""

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
include(joinpath(@__DIR__, "..", "src", "helmholtz.jl"))

using StaticArrays

# Physical constants
c = 3e8  # speed of light
f = 1.2e9  # frequency (Hz)
lambda = c / f  # wavelength

# Grid resolution
dx = lambda / 12

# Material properties
air = 1.0 + 0im
wall = 2.5 + 0.5im

# Grid size (meters)
x_meter = 10.0
y_meter = 8.0

# Grid size (pixels)
nx = Int(round(x_meter / dx))
ny = Int(round(y_meter / dx))

# Wavenumber
k = 2π * f / c

# Measurement points
measurement_points = [
    (1.0, 5.0, "Living Room"),
    (2.0, 1.0, "Kitchen"),
    (9.0, 1.0, "Bathroom"),
    (9.0, 7.0, "Bedroom")
]

function get_neighbor(rx, ry, nx, ny, dx, floor, measurement_points, step_size, trials=0)
    """
    Generates a random neighbor position within step_size distance.
    
    Returns (x_new, y_new) if valid, or (false, false) after 100 failed attempts.
    """
    if trials > 100
        return false, false
    end
    
    x_new = rx + rand() * 2 * step_size - step_size
    y_new = ry + rand() * 2 * step_size - step_size

    if check_position(x_new, y_new, nx, ny, dx, floor, measurement_points)
        return x_new, y_new    
    else 
        return get_neighbor(rx, ry, nx, ny, dx, floor, measurement_points, step_size, trials + 1)
    end
end

function simulated_annealing(rx, ry, nx, ny, dx, floor, air, wall, k, measurement_points, n_iterations, step_size, temp)
    """
    Performs simulated annealing to optimize router placement.
    
    Args:
    - rx, ry: initial position
    - n_iterations: number of iterations
    - step_size: maximum step size (meters)
    - temp: initial temperature
    
    Returns: (best_position, best_score, best_wavefield, score_history)
    """
    best = [rx, ry]
    best_eval, best_u = signal_strength(rx, ry, nx, ny, dx, floor, air, wall, k, measurement_points)
    scores = [best_eval]

    current_eval = best_eval
    current = best
    
    for i in 1:n_iterations
        t = temp / i
        
        candidate = get_neighbor(current[1], current[2], nx, ny, dx, floor, measurement_points, step_size, 0)
        if candidate == (false, false)
            println("Annealing stopped: could not find valid neighbor position.")
            return best, best_eval, best_u, scores
        end
        
        candidate_eval, candidate_u = signal_strength(candidate[1], candidate[2], nx, ny, dx, floor, air, wall, k, measurement_points)
        
        # Accept if better, or with probability based on temperature
        if candidate_eval > best_eval || rand() < exp((candidate_eval - current_eval) / t)
            current, current_eval = candidate, candidate_eval
            if candidate_eval > best_eval
                best, best_eval, best_u = candidate, candidate_eval, candidate_u
                push!(scores, best_eval)
            end
        end

        if i % 100 == 0
            println("Iteration $i, Temperature: $(round(t, digits=3)), Best: $(round(best_eval, digits=5))")
        end
    end
    
    return best, best_eval, best_u, scores 
end

using DelimitedFiles

function run_optimization_from_candidate(rx, ry, idx, floor; n_iterations=1000, step_size=0.5, start_temperature=1.0)
    println("\nStarting optimization from candidate #$idx at ($rx, $ry)")
    best, best_eval, best_u, scores = simulated_annealing(
        rx, ry, nx, ny, dx, floor, air, wall, k, measurement_points,
        n_iterations, step_size, start_temperature
    )

    println("Candidate #$idx best: ($(round(best[1], digits=3)), $(round(best[2], digits=3))) score=$(best_eval)")

    # Save best field intensity (abs(u).^2) to CSV for visualization
    intensity = abs.(best_u).^2
    writedlm("output/best_field_candidate_$(idx).csv", intensity, ',')

    return (idx=idx, init=(rx,ry), best=best, score=best_eval)
end

function main()
    println("Points per wavelength: ", round(lambda/dx))

    # Create output dir
    mkpath("output")

    # Create floor plan
    floor = create_floorplan(nx, ny, dx)

    # Read top candidates if available
    candidates = []
    topfile = "output/top_candidates.csv"
    if isfile(topfile)
        println("Reading candidates from $topfile")
        lines = readlines(topfile)
        for (i, line) in enumerate(lines)
            if i == 1
                continue # skip header
            end
            s = split(chomp(line), ',')
            if length(s) >= 3
                x = parse(Float64, s[1])
                y = parse(Float64, s[2])
                push!(candidates, (x,y))
            end
        end
    else
        println("No top_candidates.csv found, using single initial guess.")
        push!(candidates, (4.0, 5.0))
    end

    # parameters
    n_iterations = 1000
    step_size = 0.5
    start_temperature = 1.0

    results = []
    for (i, (rx, ry)) in enumerate(candidates)
        r = run_optimization_from_candidate(rx, ry, i, floor; n_iterations=n_iterations, step_size=step_size, start_temperature=start_temperature)
        push!(results, r)
    end

    # Save summary CSV
    open("output/optimization_results.csv", "w") do io
        println(io, "idx,init_x,init_y,best_x,best_y,score")
        for res in results
            println(io, string(res.idx, ",", res.init[1], ",", res.init[2], ",", res.best[1], ",", res.best[2], ",", res.score))
        end
    end

    # Print overall best
    best_overall = findmax([res.score for res in results])
    idx_best = best_overall[2]
    println("\nOverall best candidate: #$(results[idx_best].idx) start=$(results[idx_best].init) best=$(results[idx_best].best) score=$(results[idx_best].score)")
end

main()
