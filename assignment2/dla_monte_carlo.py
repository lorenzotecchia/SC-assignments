import numpy as np
import matplotlib.pyplot as plt


'''
Monte Carlo simulation of Diffusion-Limited Aggreration (DLA)

- single seed at bottom of domaion
- release random walkers from upper boundary
- walker peforms random walk until reaches cluster or walks outside system
- walker added to cluster if it reaches candidates (4-neighborhood grid)
- new random walker launched

boundaries:
- periodic boundary for left and right
- closed boundary for top and bottom
'''


NEIGHBORS = [(1, 0), (-1, 0), (0, 1), (0, -1)]


grid_len = 10
max_steps = 20
max_mass = 10
seed = 7


def launch_point(rng):
    '''
    Choose random point on the top boundary as launching point for random walker.
    '''
    i = 0
    j = rng.integers(0, grid_len)
    return(i, j)


def construct_neighborhood(cluster):
    '''
    Keeps track of the candidates in the neighborhood of the cluster.
    - candidates are empty sites in the 4-neighborhood of the occupied sites
    - include periodic and absorbing boundaries to avoid out of bound neighbors
    '''
    neighborhood = set()
    for (i, j) in cluster:
        for di, dj in NEIGHBORS:
            cand_i = i + di
            cand_j = (j + dj) % grid_len        #periodic boundary
            if 0 <= cand_i < grid_len and (cand_i, cand_j) not in cluster:
                neighborhood.add((cand_i, cand_j))
    return neighborhood


def update_neighborhood(neighborhood, added_candidate, cluster):
    '''
    If a candidate is added to the cluster, the neighborhood is updated.
    - added candidate is removed from the neighborhood
    - its empty neighbors become candidates, which are added to the neighborhood
    - include periodic and absorbing boundaries to avoid out of bound neighbors
    '''
    neighborhood.discard(added_candidate)
    i, j = added_candidate
    for di, dj in NEIGHBORS:
            cand_i = i + di
            cand_j = (j + dj) % grid_len        #periodic boundary
            if 0 <= cand_i < grid_len and (cand_i, cand_j) not in cluster:
                neighborhood.add((cand_i, cand_j))


def start_random_walker(rng, neighborhood):
    '''
    Single random walker performs random walk from launching point.
    - starts on launching point in top boundary
    - added to cluster if reaches neighborhood --> new walker
    - if random walker goes beyond the top and bottom boundary, it is removed --> new walker
    - if random walker reaches left or right boundary, it keeps going on the other side (periodic boundary)
    - if max steps reached, give up walker --> new walker
    '''
    i, j = launch_point(rng)

    for _ in range(max_steps):
        # add candidate to cluster if walker is on candidate in neighborhood
        if (i, j) in neighborhood:
            return (i, j)
        
        # otherwise keep walking randomly
        di, dj = rng.choice(NEIGHBORS)
        i += di
        j += dj

        # periodic boundary
        j = j % grid_len

        # absorbing boundary for top and bottom
        if i < 0 or i >= grid_len:
            return None 
    
    return None


def dla_simulation():
    '''
    Monte Carlo simulation for DLA on 2D grid.
    - origin in the middle of the bottom boundary
    - sends out random walkers until max mass is reached 
    '''
    rng = np.random.default_rng(seed)
    origin = (grid_len - 1, grid_len // 2)
    cluster = {origin}
    neighborhood = construct_neighborhood(cluster)

    while len(cluster) < max_mass:
        attachment = start_random_walker(rng, neighborhood)
        if attachment is None:
            continue

        # add new attachment to the cluster and update neighborhood
        cluster.add(attachment)
        update_neighborhood(neighborhood, attachment, cluster)
    return cluster


def plot_cluster(cluster):
    '''cluster_grid = np.zeros((grid_len, grid_len))
    for (i, j) in cluster:
        cluster_grid[i, j] = 1'''

    xs = [x for (x, _) in cluster]
    ys = [y for (_, y) in cluster]
    history = np.linspace(0, 1, len(cluster))

    plt.figure(figsize=(6, 6))
    plt.scatter(xs, ys, s=20, c=history, cmap='cool', vmin=0, vmax=1)
    plt.gca().set_aspect("equal", "box")
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    cluster = dla_simulation()
    plot_cluster(cluster)