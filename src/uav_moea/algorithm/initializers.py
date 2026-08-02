"""
Strategie inicjalizacji populacji dla SMS-EMOA.

1.Nearest-neighbor - zachłanne budowanie ścieżek
2.Cluster-Based - grupowanie regionów po bliskości
3.Victim-Priority - regiony z dużą ilością poszkodowanych wcześniej
4.Balanced Random - losowe ale zbalansowane długości ścieżek
5.Hybrid - mieszanka powyższych dla różnorodności populacji

"""

from __future__ import annotations

import random

from uav_moea.io.data_loader import TestCase
from uav_moea.model import Solution


#------Nearest Neighbor-----------------------

def _nearest_neighbor_path(
        start: int,
        available: set[int],
        test_case: TestCase,
        max_regions: int | None = None,

) -> list[int]:
    """Buduj ścieżkę zachłannie - zawsze leć do najbliższego regionu"""
    path = [start]
    available.discard(start)
    current = start

    while available:
        if max_regions and len(path) >= max_regions:
            break

        nearest = min(available, key=lambda r: test_case.get_flight_time(current, r))
        path.append(nearest)
        available.discard(nearest)
        current = nearest

    return path

def generate_nearest_neighbor(test_case: TestCase, num_uavs: int) -> Solution:
    """Każdy dron startuje z regionu o najkrótszym reach time, potem zachłannie odwiedza kolejne regiony
    
    """

    remaining = set(range(test_case.node_count))
    regions_per_uav = test_case.node_count // num_uavs
    sorted_by_reach = sorted(remaining, key=lambda r: test_case.get_reach_time(r))

    paths = []
    for i in range(num_uavs):
        if not remaining:
            paths.append([])
            continue

        # Najlepszy dostępny start
        start = next(r for r in sorted_by_reach if r in remaining)

        max_r = None if i == num_uavs - 1 else regions_per_uav

        path = _nearest_neighbor_path(start, remaining, test_case, max_regions=max_r)
        paths.append(path)

    return Solution(paths=paths, num_uavs=num_uavs)


#------------------Cluster-Based----------------------------------


def generate_cluster_based(test_case: TestCase, num_uavs: int) -> Solution:
    """
    Grupujemy regiony po reach time przypisujemy klastry do dronów

    """

    sorted_regions =sorted(
        range(test_case.node_count),
        key=lambda r: test_case.get_reach_time(r),
    )

    # Podział na klastry
    chunk = test_case.node_count // num_uavs
    clusters = []

    for i in range(num_uavs):
        if i < num_uavs - 1:
            clusters.append(sorted_regions[i * chunk : (i+1) * chunk])
        else:
            clusters.append(sorted_regions[i*chunk:])

    random.shuffle(clusters)

    # Uporządkowanie każdego klastra
    paths = []
    for cluster in clusters:
        if not cluster:
            paths.append([])
            continue

        available = set(cluster)
        start = min(cluster, key=lambda r: test_case.get_reach_time(r))
        path = _nearest_neighbor_path(start, available, test_case)
        paths.append(path)

    return Solution(paths=paths, num_uavs=num_uavs)
    

        
#---------Victim-Prioririty------------------------

def generate_victim_priority(test_case: TestCase, num_uavs: int) -> Solution:
    """
    Regiony z największą ilością poszkodowanych rozkładane na round-robin miedzy drony, potem uporządkowane po nearest neighbor

    """

    by_victims = sorted(
        range(test_case.node_count),
        key=lambda r: test_case.get_population(r),
        reverse=True
    )

    region_sets = [[] for _ in range(num_uavs)]
    for idx, region in enumerate(by_victims):
        region_sets[idx % num_uavs].append(region)

    # Kolejność: najpierw regiony z największą liczbą ofiar
    paths = []
    for regions in region_sets:
        path = sorted(regions, key=lambda r: test_case.get_population(r), reverse=True)
        paths.append(path)

    return Solution(paths=paths, num_uavs=num_uavs)


#----------Balanced-Random-----------------

def generate_balanced_random(test_case : TestCase, num_uavs : int) -> Solution:
    """Losowy podzial regionow ale zbalansowane dlugosci (+/- 1)"""

    regions = list(range(test_case.node_count))
    random.shuffle(regions)

    paths = [[] for _ in range(num_uavs)]
    for idx, region in enumerate(regions):
        paths[idx % num_uavs].append(region)

    for path in paths:
        random.shuffle(path)


    return Solution(paths=paths, num_uavs=num_uavs)


#------------Hybrid------------------------

def _perturb(solution : Solution, test_case: TestCase, n_swaps: int = 10) -> Solution:
    """Kilka losowyh swapów dla różnorodności heurystycznych rozwiązań"""
    sol = solution.copy()

    for _ in range(n_swaps):
        candidates = [i for i, p in enumerate(sol.paths) if len(p) >= 2]
        if not candidates:
            break
        uav_id = random.choice(candidates)
        path = sol.paths[uav_id]
        pos1, pos2 = random.sample(range(len(path)), 2)
        path[pos1], path[pos2] = path[pos2], path[pos1]

    if not sol.validate(test_case):
        return solution
    
    return sol


def initialize_population(
    test_case : TestCase,
    num_uavs: int,
    population_size : int,
    nn_fraction: float = 0.2,
    cluster_fraction: float = 0.1,
    victim_fraction: float = 0.1,                 
) -> list[Solution]:
    """
    Generujemy zróżnicowaną populację startową za pomocą różnych podejść do inicjalizacji
    
    Domyślny podział: Nearest-neighbor: 20%, Cluster: 10%, Victim-prio: 10%, Balanced Random: 60%
    
    """
    nn_count = max(1, int(population_size * nn_fraction))
    cluster_count = max(1, int(population_size * cluster_fraction))
    victim_count = max(1, int(population_size * victim_fraction))
    balanced_count = population_size - nn_count - cluster_count - victim_count

    population = []

    for _ in range(nn_count):
        sol = generate_nearest_neighbor(test_case, num_uavs)
        population.append(_perturb(sol, test_case))

    for _ in range(cluster_count):
        sol = generate_cluster_based(test_case, num_uavs)
        population.append(_perturb(sol, test_case))

    for _ in range(victim_count):
        sol = generate_victim_priority(test_case, num_uavs)
        population.append(_perturb(sol, test_case))

    for _ in range(balanced_count):
        sol = generate_balanced_random(test_case, num_uavs)
        population.append(sol)  # random nie potrzebuje perturbacji

    return population
