"""
Coarse grid evaluation script for WiFi router placement.

This script:
1. Defines a search region (polygon)
2. Creates a coarse grid of candidate positions
3. Evaluates signal strength at each candidate
4. Saves results to CSV files for visualization in Python
"""

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
include(joinpath(@__DIR__, "..", "src", "helmholtz.jl"))

using StaticArrays
using PolygonOps
using DelimitedFiles
using Printf

# Physical constants
c = 3e8  # speed of light
f = 1.2e9  # frequency (Hz)
lambda = c / f  # wavelength

# Grid resolution (points per wavelength)
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

# Measurement points (x, y, name)
measurement_points = [
    (1.0, 5.0, "Living Room"),
    (2.0, 1.0, "Kitchen"),
    (9.0, 1.0, "Bathroom"),
    (9.0, 7.0, "Bedroom")
]

function coarse_candidate_evaluation(inside, points, spacing, M, nx, ny, dx, floor, measurement_points)
    """
    Evaluates signal strength on a coarse grid of candidate positions.
    
    Returns:
    - scores: matrix of signal strengths (NaN for invalid positions)
    """
    candidate_idxs = findall(inside)
    total = length(candidate_idxs)

    if total == 0
        println("No candidates inside polygon. Exiting.")
        return
    end

    num_x = Int(maximum(getindex.(points, 1)) / spacing) + 1
    num_y = Int(maximum(getindex.(points, 2)) / spacing) + 1
    
    scores = fill(NaN, num_x, num_y)
    count = 0

    for ip in candidate_idxs
        count +=1
        x, y = points[ip]
        ix = Int(round(x / spacing)) + 1
        iy = Int(round(y / spacing)) + 1
        
        if check_position(x, y, nx, ny, dx, floor, measurement_points)
            scores[ix, iy], _ = signal_strength(M, x, y, nx, ny, dx, measurement_points)
        end

        # print percentage progress every 1% or on last
        pct = Int(round(100 * count / total))
        if count == total || pct % 1 == 0
            @printf("\rProgress: %3d%% (%d/%d)", pct, count, total)
            flush(stdout)
        end
    end

    return scores    
end

function top_n_points(scores, spacing; n=10)
    # Collect valid indices and values
    inds = CartesianIndices(scores)
    vals = [(I, scores[I]) for I in inds if !isnan(scores[I])]
    # sort descending by score
    sort!(vals, by = x -> -x[2])
    m = min(n, length(vals))
    res = Vector{Tuple{Float64,Float64,Float64}}()
    for k in 1:m
        I, v = vals[k]
        x_m = (I[1]-1) * spacing
        y_m = (I[2]-1) * spacing
        push!(res, (x_m, y_m, v))
    end
    return res
end

function save_top_n(scores, spacing, n, filename)
    top = top_n_points(scores, spacing; n=n)
    open(filename, "w") do io
        println(io, "x,y,score")
        for (x,y,s) in top
            println(io, string(x, ",", y, ",", s))
        end
    end
end

function main()
    println("Points per wavelength: ", round(lambda/dx))
    
    # Create floor plan and helmholtz matrix
    floor = create_floorplan(nx, ny, dx)
    M = build_helmholtz_matrix(nx, ny, dx, floor, air, wall, k)
    F = lu(M)

    # Define search region (polygon vertices)
    x_poly = [2., 2., 1., 1., 9., 9., 7.5, 7.5, 2.]
    y_poly = [0., 1.5, 1.5, 6.5, 6.5, 3., 3., 0., 0.]
    polygon = SVector.(x_poly, y_poly)
    
    # Coarse grid spacing (meters)
    spacing = 0.5
    
    # Generate candidate grid
    xa = 0:spacing:x_meter
    ya = 0:spacing:y_meter
    points = vec([SVector(x, y) for x in xa, y in ya])
    
    # Check which points are inside the search region
    inside = [inpolygon(p, polygon; in=true, on=false, out=false) for p in points]
    println("Total candidate positions: ", sum(inside))
    
    # Evaluate signal strength on coarse grid (iterate only over inside points)
    # scores = fill(NaN, Int(maximum(getindex.(points,1)) / spacing) + 1, Int(maximum(getindex.(points,2)) / spacing) + 1)
    scores = coarse_candidate_evaluation(inside, points, spacing, F, nx, ny, dx, floor, measurement_points)

    println()    
    
    # Save results to CSV
    println("\nSaving results...")
    # create output dir if missing
    mkpath("output")
    writedlm("output/coarse_scores.csv", scores, ',')
    writedlm("output/floor_plan.csv", floor, ',')

    # Save top-n candidate positions
    save_top_n(scores, spacing, 10, "output/top_candidates.csv")
    println("Saved coarse_scores.csv, floor_plan.csv and top_candidates.csv to output/")
    println("Grid spacing: $spacing meters")
    println("Score matrix size: $(size(scores))")
end

main()
