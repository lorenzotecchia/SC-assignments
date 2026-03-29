"""
Core Helmholtz equation solver for WiFi signal propagation.

This module provides the fundamental functions for:
- Creating apartment floor plans
- Building and solving the Helmholtz equation
- Computing signal strength at measurement points
- Position validation for router placement
"""

using SparseArrays
using LinearAlgebra
using Random
using Statistics

# Coordinate conversion: meters to grid indices
function m2x(x, dx)
    return Int(round(x/dx)) + 1
end
function m2y(y, dx) 
    return Int(round(y/dx)) + 1
end

function create_floorplan(nx, ny, dx)
    """
    Creates the apartment floor plan matrix.
    
    Returns:
    - floor matrix where air = 1, wall = 0
    """
    floor = ones(Int, nx, ny)
    m2p_x(x) = clamp(Int(round(x / dx)) + 1, 1, nx)
    m2p_y(y) = clamp(Int(round(y / dx)) + 1, 1, ny)

    t_wall = 0.15
    t_pixel =max(1, Int(round(t_wall / dx)))
    t_half_pixel = max(1, Int(round(t_wall *0.5 / dx)))

    floor[1:t_pixel, :] .= 0
    floor[end-t_pixel+1:end, :] .= 0
    floor[:, 1:t_pixel] .= 0
    floor[:, end-t_pixel+1:end] .= 0

    y = m2p_y(3.0)
    floor[:, y-t_half_pixel:y+t_half_pixel-1] .= 0

    floor[m2p_x(3.0)+1:m2p_x(4.0)-1, y-t_half_pixel:y+t_half_pixel-1] .= 1
    floor[m2p_x(6.0)+1:m2p_x(7.0)-1, y-t_half_pixel:y+t_half_pixel-1] .= 1

    x = m2p_x(2.5)
    floor[x-t_half_pixel:x+t_half_pixel-1, m2p_y(0.0):m2p_y(2.0)] .= 0

    x = m2p_x(6.0)
    floor[x-t_half_pixel:x+t_half_pixel-1, m2p_y(3.0):m2p_y(8.0)] .= 0

    x = m2p_x(7.0)
    floor[x-t_half_pixel:x+t_half_pixel-1, m2p_y(0.0):m2p_y(1.5)] .= 0
    floor[x-t_half_pixel:x+t_half_pixel-1, m2p_y(2.5):m2p_y(3.0)] .= 0

    return floor
end


function build_helmholtz_matrix(nx, ny, dx, floor, air, wall, k)
    """
    Builds sparse Helmholtz matrix M for the 2D domain.
    
    Discretization: 5-point stencil finite differences
    Boundary conditions: Sommerfeld radiation conditions
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
        n = floor[x,y] == 1 ? air : wall

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


function gaussian_source(rx, ry, nx, ny, dx; sigma = 0.2, A = 0.2)
    """
    Returns Wifi router source term f as Gaussian pulse. 
    """

    f = zeros(ComplexF64, nx, ny)

    # router location
    router_x = m2x(rx, dx)
    router_y = m2y(ry, dx)

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


function solve_Helmholtz_eq(M, rx, ry, nx, ny, dx)
    """
    Solves the Helmholtz equation for router at (rx, ry).
    
    Returns wave field u as a 2D array.
    """
    f = gaussian_source(rx, ry, nx, ny, dx)

    # compute u = M^-1 * f and reshape to 2D
    u = reshape(M \ f, nx, ny)

    return u
end

function diagnose(u, floor)
    """
    Diagnostic function to analyze signal intensity distribution.
    """
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

function signal_strength(M, rx, ry, nx, ny, dx, measurement_points; verbose=false)
    """
    Computes total signal strength at all measurement points for router at (rx, ry).
    
    Returns: (total_signal, wave_field)
    """
    u = solve_Helmholtz_eq(M, rx, ry, nx, ny, dx)

    sum = 0.0
    for (mx, my, name) in measurement_points
        signal = measurement(u, nx, ny, dx, mx, my)
        if verbose
            println("Signal in ", name, ": ", signal)
        end
        sum += signal
    end
    if verbose
        println("Total signal: ", sum)
    end
    return sum, u
end

function measurement(u, nx, ny, dx, mx, my)
    """
    Measures signal intensity in a 5cm radius region around point (mx, my).
    """
    ix = m2x(mx, dx)
    iy = m2y(my, dx)

    radius = max(1, Int(round(0.05 / dx)))

    total = 0.
    for i in -radius:radius
        for j in -radius:radius
            if i^2+j^2<radius^2
                x = ix + i 
                y = iy + j 
                if 1 <= x <= nx && 1 <= y <= ny 
                    total += abs(u[x,y])^2
                end
            end
        end
    end
    return total * dx^2
end

function check_position(rx, ry, nx, ny, dx, floor, measurement_points)
    """
    Validates if router position (rx, ry) is valid.
    
    Criteria:
    - Must be within bounds
    - Must be in air (not wall)
    - Must be at least 0.5m away from measurement points
    """
    ix = m2x(rx, dx)
    iy = m2y(ry, dx)

    # Check bounds
    if ix <= 1 || iy <= 1 || ix >= nx || iy >= ny
        return false 
    end
    
    # Check not in wall
    if floor[ix, iy] == 0
        return false
    end

    # Check distance from measurement points
    for (mx, my, _) in measurement_points
        if sqrt((mx-rx)^2 + (my-ry)^2) <= 0.5
            return false
        end
    end
    
    return true
end