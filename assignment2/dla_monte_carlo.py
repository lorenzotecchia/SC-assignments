import numpy as np


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

grid size: 100 x 100
'''


NEIGHBORS = [(1, 0), (-1, 0), (0, 1), (0, -1)]

grid_len = 100
max_steps = 10_000
max_mass = 2000


def launch_point(rng):
    '''
    Choose random point on the top boundary as launching point for random walker.
    '''
    i = 0
    j = rng.integers(0, grid_len)
    return(i, j)


def neighborhood(cluster):
    '''
    Keeps track of the candidates in the neighborhood of the cluster.
    Candidates are empty sites in the 4-neighborhood of the occupied sites.
    '''
    candidates = set()
    for (i, j) in cluster:
        for di, dj in NEIGHBORS:
            cand = (i + di, j + dj)
            if cand not in cluster:
                candidates.add(cand)
    return candidates


def update_neighborhood(candidates, added_candidate, cluster):
    '''
    If a candidate is added to the cluster, the neighborhood is updated.
    - added candidate is removed from the neighborhood
    - its empty neighbors become candidates, which are added to the neighborhood
    '''
    candidates.discard(added_candidate)
    i, j = added_candidate
    for di, dj in NEIGHBORS:
        cand = (i + di, j + dj)
        if cand not in cluster:
            candidates.add(cand)


def random_walker(rng, neighborhood):
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

    


        