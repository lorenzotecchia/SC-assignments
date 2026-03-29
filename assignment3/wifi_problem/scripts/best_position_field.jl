push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
include(joinpath(@__DIR__, "..", "src", "helmholtz.jl"))

using DelimitedFiles

# Physical constants
c = 3e8
f = 2.4e9
lambda = c / f
dx = lambda / 12

air  = 1.0 + 0im
wall = 2.5 + 0.5im

x_meter, y_meter = 10.0, 8.0
nx = Int(round(x_meter / dx))
ny = Int(round(y_meter / dx))
k  = 2π * f / c

# --- Read best position (rank 1) from top100 CSV ---
infile = "output/top100_annealing.csv"
if !isfile(infile)
    println("ERROR: $infile not found."); exit(1)
end

lines = readlines(infile)
if length(lines) < 2
    println("No data in $infile"); exit(1)
end

p  = split(strip(lines[2]), ",")
rx = parse(Float64, strip(p[2]))
ry = parse(Float64, strip(p[3]))

if isnan(rx)
    println("ERROR: rank-1 entry not found in $infile"); exit(1)
end
println("Best position: rx=$rx, ry=$ry")

# --- Solve ---
floor_plan = create_floorplan(nx, ny, dx)
M          = build_helmholtz_matrix(nx, ny, dx, floor_plan, air, wall, k)
u_best     = solve_Helmholtz_eq(M, rx, ry, nx, ny, dx)

# --- Save intensity (real-valued) ---
intensity = abs.(u_best).^2
writedlm("output/best_field_overall.csv", intensity, ',')
println("Saved overall best field to output/best_field_overall.csv")