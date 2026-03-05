using Random
using Plots

#=
Monte Carlo simulation of Diffusion-Limited Aggreration (DLA)

- single seed at bottom of domain
- release random walkers from upper boundary
- walker peforms random walk until reaches cluster or walks outside system
- walker added to cluster if it reaches candidates (4-neighborhood grid)
- new random walker launched

boundaries:
- periodic boundary for left and right
- closed boundary for top and bottom

grid size: 100 x 100
=#


NEIGHBORS = [(1, 0), (-1, 0), (0, 1), (0, -1)]

grid_len = 100
max_steps = 10000
max_mass = 400
seed = 7


function launch_point(rng)
    #= Choose random point on the top boundary as launching point for random walker. =#

    j = rand(rng, 1:grid_len)
    return (1, j)
end


function construct_neighborhood(cluster)
    #= Keeps track of the candidates in the neighborhood of the cluster.
    - candidates are empty sites in the 4-neighborhood of the occupied sites
    - include periodic and absorbing boundaries to avoid out of bound neighbors =#
    
    neighborhood = Set{Tuple{Int, Int}}()
    for (i, j) in cluster
        for (di, dj) in NEIGHBORS
            cand_i = i + di
            cand_j = ((j - 1 + dj) % grid_len) + 1 #periodic boundary
            if 0 < cand_i <= grid_len && !in((cand_i, cand_j), cluster)
                push!(neighborhood, (cand_i, cand_j))
            end
        end
    end
    return neighborhood
end


function update_neighborhood(neighborhood, added_candidate, cluster)
    #= If a candidate is added to the cluster, the neighborhood is updated.
    - added candidate is removed from the neighborhood
    - its empty neighbors become candidates, which are added to the neighborhood
    - include periodic and absorbing boundaries to avoid out of bound neighbors =#

    pop!(neighborhood, added_candidate)
    i, j = added_candidate
    for (di, dj) in NEIGHBORS
        cand_i = i + di
        cand_j = ((j - 1 + dj) % grid_len) + 1 #periodic boundary
        if 0 < cand_i <= grid_len && !in((cand_i, cand_j), cluster)
            push!(neighborhood, (cand_i, cand_j))
        end
    end
end


function release_random_walker(rng, neighborhood)
    #= Single random walker performs random walk from launching point.
    - starts on launching point in top boundary
    - added to cluster if reaches neighborhood --> new walker
    - if random walker goes beyond the top and bottom boundary, it is removed --> new walker
    - if random walker reaches left or right boundary, it keeps going on the other side (periodic boundary)
    - if max steps reached, give up walker --> new walker =#

    i, j = launch_point(rng)

    for _ in 1:max_steps
        # check if walker is on site in neighborhood
        if in((i, j), neighborhood)
            return (i, j)
        end
        
        (di, dj) = NEIGHBORS[rand(rng, 1:4)]
        i += di
        j += dj

        # periodic boundary for left and right
        j = (j - 1) % grid_len + 1

        # absorbing boundary for top and bottom
        if i < 1 || i > grid_len
            return nothing
        end
    end
    return nothing
end


function release_random_walker_ps(rng, neighborhood, ps, cluster)
    #= Single random walker performs random walk with sticking probability ps.
    - starts on launching point in top boundary
    - added to cluster if reaches neighborhood with probability ps--> new walker
    - cannot walk through cluster sites
    - if random walker goes beyond the top and bottom boundary, it is removed --> new walker
    - if random walker reaches left or right boundary, it keeps going on the other side (periodic boundary)
    - if max steps reached, give up walker --> new walker =#

    i, j = launch_point(rng)

    for _ in 1:max_steps
        # check if walker is on site in neighborhood
        if in((i, j), neighborhood)
            # attach to cluster with probability ps
            if rand(rng) < ps
                return (i, j)
            end
        end
        
        (di, dj) = NEIGHBORS[rand(rng, 1:4)]
        step_i = i + di
        step_j = j + dj

        # walker cannot walk through cluster
        if in((step_i, step_j), cluster)
            continue
        end
        i, j = step_i, step_j

        # periodic boundary for left and right
        j = (j - 1) % grid_len + 1

        # absorbing boundary for top and bottom
        if i < 1 || i > grid_len
            return nothing
        end
    end
    return nothing
end


function dla_simulation()
    #= Monte Carlo simulation for DLA on 2D grid.
    - origin in the middle of the bottom boundary
    - sends out random walkers until max mass is reached =#

    rng = MersenneTwister(seed)
    origin = (grid_len, div(grid_len, 2))
    cluster = Set{Tuple{Int, Int}}((origin,))
    neighborhood = construct_neighborhood(cluster)

    while length(cluster) < max_mass
        attachment = release_random_walker(rng, neighborhood)
        if attachment !== nothing
            push!(cluster, attachment)
            update_neighborhood(neighborhood, attachment, cluster)
        end
    end
    return cluster
end


function dla_simulation_ps(ps)
    #= Monte Carlo simulation for DLA on 2D grid with sticking probability ps.
    - origin in the middle of the bottom boundary
    - sends out random walkers until max mass is reached =#

    rng = MersenneTwister(seed)
    origin = (grid_len, div(grid_len, 2))
    cluster = Set{Tuple{Int, Int}}((origin,))
    neighborhood = construct_neighborhood(cluster)

    while length(cluster) < max_mass
        attachment = release_random_walker_ps(rng, neighborhood, ps, cluster)
        if attachment !== nothing
            push!(cluster, attachment)
            update_neighborhood(neighborhood, attachment, cluster)
        end
    end
    return cluster
end


function plot_cluster(cluster)
    #= Plots the cluster with origin at bottom
    - swaps x and y 
    - inverts y-axis =#

    x = [i for (i, _) in cluster]
    y = [j for (_, j) in cluster]
    
    # x and y swaped
    scatter(y, x, marker=:circle, color=:blue, size=(600, 600), axis=false, legend=false)
    
    # Invert y-axis 
    plot!(yflip=true)
    
    # Make dots evenly spaced
    plot!(aspect_ratio=1)
end


function plot_clusters_ps(probabilities)
    pl = plot(layout = (1,3), size=(1200, 900))
    for (i, ps) in enumerate(probabilities)
        cluster = dla_simulation_ps(ps)
        
        x = [i for (i, _) in cluster]
        y = [j for (_, j) in cluster]
        
        # plot in the i-th subplot
        scatter!(pl[i], y, x, marker=:circle, color=:blue, axis=false, legend=false)
        
        # invert the y-axis for the plot to have the origin at the bottom
        plot!(pl[i], yflip=true, title="Sticking Probability = $ps")
        
        # ensure the dots are evenly spaced (aspect ratio)
        plot!(pl[i], aspect_ratio=1)

        #title!(pl[i], "Sticking Probability = $ps")
    end
    display(pl)
end


#cluster = dla_simulation()
#plot_cluster(cluster)

# plot clusters with different ps
probabilities = [0.1, 0.5, 1.0]
plot_clusters_ps(probabilities)
