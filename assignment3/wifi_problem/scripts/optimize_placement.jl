"""
Simulated annealing optimization for WiFi router placement.

This script uses simulated annealing to find the optimal router position
that maximizes total signal strength at all measurement points.
"""

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
include(joinpath(@__DIR__, "..", "src", "helmholtz.jl"))

using StaticArrays
using Printf

# Physical constants
c = 3e8  # speed of light
f = 2.4e9  # frequency (Hz)
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
        return [x_new, y_new]   
    else 
        return get_neighbor(rx, ry, nx, ny, dx, floor, measurement_points, step_size, trials + 1)
    end
end

function simulated_annealing(M, rx, ry, nx, ny, dx, floor, measurement_points, n_iterations, step_size, temp; output_path = false, topk)
    """
    Performs simulated annealing to optimize router placement.

    Args:
    - rx, ry: initial position
    - n_iterations: number of iterations
    - step_size: maximum step size (meters)
    - temp: initial temperature

    Returns: (best_position, best_score, best_wavefield, score_history) or with path when output_path=true
    """
    best = [rx, ry]

    # initial solve
    best_eval, best_u = signal_strength(M, rx, ry, nx, ny, dx, measurement_points)

    scores = [best_eval]

    if !isnothing(topk)
        insert!(topk, best_eval, rx, ry)
    end

    # path storage (vector of [x,y,score] vectors) when requested
    path = Vector{Vector{Float64}}()
    if output_path
        push!(path, [best[1], best[2], best_eval])
    end

    current_eval = best_eval
    current = best

    for i in 1:n_iterations
        t = temp / i

        candidate = get_neighbor(current[1], current[2], nx, ny, dx, floor, measurement_points, step_size, 0)
        if candidate == (false, false)
            println("Annealing stopped: could not find valid neighbor position.")
            return best, best_eval, best_u, scores
        end

        # evaluate candidate by solving with prebuilt M
        candidate_eval, candidate_u = signal_strength(M, candidate[1], candidate[2], nx, ny, dx, measurement_points)


        if !isnothing(topk)
            insert!(topk, candidate_eval, candidate[1], candidate[2])
        end

        # Accept if better, or with probability based on temperature
        if candidate_eval > best_eval || rand() < exp((candidate_eval - current_eval) / t)
            current, current_eval = candidate, candidate_eval
            if output_path
                push!(path, [current[1], current[2], current_eval])
            end
            if candidate_eval > best_eval
                best, best_eval, best_u = candidate, candidate_eval, candidate_u
                push!(scores, best_eval)
            end
        end

        # print percentage progress every 1% or on last
        pct = Int(round(100 * i / n_iterations))
        if i == n_iterations || pct % 1 == 0
            @printf("\rProgress: %3d%% (%d/%d)", pct, i,n_iterations)
            flush(stdout)
        end

    end

    if output_path
        return best, best_eval, best_u, scores, path
    end
    return best, best_eval, best_u, scores
end

using DelimitedFiles

function run_optimization(rx, ry, M, floor, idx; n_iterations=1000, step_size=0.5, start_temperature=1.0, output_path=false, topk=nothing)
    println("\nStarting optimization from candidate #$idx at ($rx, $ry), with step size $step_size and starting temperature $start_temperature")
    if output_path
        best, best_eval, best_u, scores, path = simulated_annealing(
            M, rx, ry, nx, ny, dx, floor, measurement_points,
            n_iterations, step_size, start_temperature; output_path=true, topk=topk
        )
    else
        best, best_eval, best_u, scores = simulated_annealing(
            M, rx, ry, nx, ny, dx, floor, measurement_points,
            n_iterations, step_size, start_temperature; output_path=false, topk=topk
        )
    end

    println("Candidate #$idx best: ($(round(best[1], digits=3)), $(round(best[2], digits=3))) score=$(best_eval)")

    # Save best field intensity (abs(u).^2) to CSV for visualization
    intensity = abs.(best_u).^2
    writedlm("output/best_field_candidate_$(idx).csv", intensity, ',')

    # If requested, save the path taken (x,y,score) per row
    if output_path && length(path) > 0
        println("acceptance rate: ", round(length(path)/n_iterations),"%")
        path_mat = transpose(hcat(path...))
        open("output/optimization_path_candidate_$(idx).csv", "w") do io
            println(io, "x,y,score")
            for r in 1:size(path_mat,1)
                println(io, string(path_mat[r,1], ",", path_mat[r,2], ",", path_mat[r,3]))
            end
        end
    end

    return (idx=idx, init=(rx,ry), best=best, score=best_eval)
end

# Add this struct and helpers at the top of the file
mutable struct TopK
    k::Int
    entries::Vector{Tuple{Float64, Float64, Float64}}  # (score, x, y)
end

TopK(k) = TopK(k, Tuple{Float64, Float64, Float64}[])

function insert!(topk::TopK, score, x, y)
    push!(topk.entries, (score, x, y))
    sort!(topk.entries, by=e->e[1], rev=true)
    if length(topk.entries) > topk.k
        resize!(topk.entries, topk.k)
    end
end

function save_topk(topk::TopK, path="output/top100_annealing.csv")
    open(path, "w") do io
        println(io, "rank,x,y,score")
        for (i, (score, x, y)) in enumerate(topk.entries)
            println(io, "$i,$(round(x,digits=4)),$(round(y,digits=4)),$(round(score,digits=6))")
        end
    end
end

function main()
    topk = TopK(100)

    println("Points per wavelength: ", round(lambda/dx))

    # Create output dir
    mkpath("output")

    # Create floor plan
    floor = create_floorplan(nx, ny, dx)
    M = build_helmholtz_matrix(nx, ny, dx, floor, air, wall, k)
    F = lu(M)

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
    step_size = 0.15
    start_temperature = 0.5

    results = []
    for (i, (rx, ry)) in enumerate(candidates)
        r = run_optimization(rx, ry, F, floor, i; n_iterations=n_iterations, step_size=step_size, start_temperature=start_temperature, output_path=true, topk=topk)
        push!(results, r)
    end
    # Save summary CSV
    open("output/optimization_results.csv", "w") do io
        println(io, "idx,init_x,init_y,best_x,best_y,score")
        for res in results
            println(io, string(res.idx, ",", res.init[1], ",", res.init[2], ",", res.best[1], ",", res.best[2], ",", res.score))
        end
    end
    save_topk(topk)
    println("Saved top-$(topk.k) positions to output/top100_annealing.csv")


    # Path files are saved per-candidate inside run_optimization_from_candidate when requested
    # (output_path flag).

    # Print overall best
    best_overall = findmax([res.score for res in results])
    idx_best = best_overall[2]
    println("\nOverall best candidate: #$(results[idx_best].idx) start=$(results[idx_best].init) best=$(results[idx_best].best) score=$(results[idx_best].score)")

end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
