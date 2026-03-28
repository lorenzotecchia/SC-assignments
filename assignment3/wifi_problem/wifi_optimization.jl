
"""
1. implement distanced-based decay (pathloss)
2. implement wall attenuation (shadowing) 
3. combine signal
4. simulated annealing
"""


function pathloss(router, dim_x, dim_y, scaling)
    """
    Computes the distance to the router and converts distance into signal loss
    - returns signal loss matrix  
    - uses free-space propagation law
    """

    # store distance from router for each pixel
    Dist = zeros(Float64, dim_x, dim_y)

    # compute distance
    for x in 1:dim_x, y in 1:dim_y
        Dist[x,y] = abs(complex(router...) - complex(x,y)) * scaling
    end

    # fix singularity
    Dist[router...] = Dist[router[1]-1, router[2]-1]

    # compute path loss
    sigal_loss = similar(Dist, Float64)
    sigal_loss = 20 .* log10.(Dist .^ -2)
    return sigal_loss
end


function pixels_on_line(x0, x1, y0, y1)
    """
    Returns all pixel along a straight line between two points (x_0, y_0) and (x_1, y_1). 
    - swap axis if line is steep to ensure we step along the longer direction: uniform sampling
    """
    # store final list of pixels along line
    result = Array{Tuple{Int, Int}}(undef, 0)

    # transformation function
    rev = identity

    #check slope of line: if steep line, swap axis
    if abs(y1-y0) ≤ abs(x1-x0)
        x0, y0, x1, y1 = y0, x0, y1, x1

        # undo swap later
        rev = reverse
    end

    len_y = abs(y1 - y0)

    for i in 0:len_y
        push!(result, rev((round(Int, i//len_y * (x1-x0) + x0), (y1>y0 ? 1 : -1)*i + y0)))
    end

    return result
end


function shadowing_wave(floor, router)
    """
    Computes 2D matrix S of signal strength at each pixel using shadowing based on complex refractive indices.
    Based on:
    - phase shift and attenuation
    - distance from router
    """

    nx, ny = size(floor)

    # refractive index
    index_air = 1.0 + 0im
    index_wall = 2.5 + 0.5im

    # refractive index field: matrix with complex numbers
    index_field = Array{ComplexF64}(undef, nx, ny)
    for x in 1:nx, y in 1:ny
        index_field[x,y] = (floor[x,y] == 1) ? index_air : index_wall
    end

    # store signal strength per pixel
    S = zeros(Float64, nx, ny)

    # compute signal strength from each pixel
    for y in 1:ny, x in 1:nx
        # ray propagation (straight line)
        pixels = pixels_on_line(router[1], router[2], x, y)

        # wave variables
        phase_shift = 0.0       # wave oscillation
        attenuation = 1.0       # singal strength multiplier

        for (px, py) in pixels
            # material property of pixel
            property = index_field[px, py]

            # phase shift
            phase_shift += real(property)

            # attenuation (exponential decay)
            attenuation *= exp(-imag(property))
        end
        
        # combine effects to get signal strength
        S[x,y] = attenuation * cos(phase_shift)
    end

    return S
end