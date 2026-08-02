"""
Operatory genetyczne: krzyżowanie i mutacje ścieżek UAV.

Crossover: Order Crossover (OX) adaptowany dla multi-UAV

Mutacje wewnątrz ścieżki (intra-route):
  - swap    — zamiana dwóch losowych regionów w ścieżce
  - insert  — wyjmij region i wstaw w inne miejsce (relocate)
  - invert  — odwróć losowy segment (2-opt intra)
  - or_opt  — przenieś spójny blok k regionów w inne miejsce tej samej ścieżki
              lub do innej ścieżki (Or-opt, k∈{2,3})

Mutacje między ścieżkami (inter-route):
  - transfer    — przenieś jeden region z jednej ścieżki do drugiej
  - exchange    — zamień jeden region między dwiema ścieżkami (1:1)
  - cross_route — 2-opt*: zamień ogony dwóch ścieżek za losowym punktem podziału

Literatura:
  Prins (2004) Computers & OR — OX + operatory VRP dla chromosomu gigantycznego
  Or (1976) — Or-opt jako uogólnienie 2-opt przez przenoszenie bloków
  Potvin & Rousseau (1995) — 2-opt* (cross-route) dla m-TSP/VRP
"""

from __future__ import annotations

import random

from uav_moea.io.data_loader import TestCase
from uav_moea.model.individual import Solution


# ── Order Crossover ───────────────────────────────────────────


def order_crossover(parent1: list[int], parent2: list[int]) -> list[int]:
    """OX dla pojedynczej permutacji."""
    size = len(parent1)
    cut1, cut2 = sorted(random.sample(range(size), 2))

    offspring = [None] * size
    offspring[cut1:cut2] = parent1[cut1:cut2]

    segment = set(offspring[cut1:cut2])
    fill = [x for x in parent2 if x not in segment]

    idx = cut2
    for gene in fill:
        if idx >= size:
            idx = 0
        while offspring[idx] is not None:
            idx += 1
            if idx >= size:
                idx = 0
        offspring[idx] = gene
        idx += 1

    return offspring


def crossover(parent1: Solution, parent2: Solution, test_case: TestCase) -> Solution:
    """
    Order Crossover dla wielu UAV.

    1. CONCAT  — połącz ścieżki w jedną permutację (chromosom gigantyczny)
    2. OX      — zastosuj Order Crossover na całej permutacji
    3. RESTORE — podziel z powrotem do k ścieżek używając punktów podziału
                 losowo wybranego rodzica
    """
    full1, splits1 = [], []
    for path in parent1.paths:
        full1.extend(path)
        splits1.append(len(full1))

    full2, splits2 = [], []
    for path in parent2.paths:
        full2.extend(path)
        splits2.append(len(full2))

    offspring_full = order_crossover(full1, full2)

    splits = random.choice([splits1, splits2])
    paths = []
    prev = 0
    for sp in splits:
        paths.append(offspring_full[prev:sp])
        prev = sp

    offspring = Solution(paths=paths, num_uavs=parent1.num_uavs)

    if not offspring.validate(test_case):
        return parent1.copy()

    return offspring


# ── Helpery ───────────────────────────────────────────────────


def _pick_path(solution: Solution) -> tuple[int, list[int]]:
    """Wybierz losową ścieżkę drona (indeks, ścieżka)."""
    uav_id = random.randint(0, solution.num_uavs - 1)
    return uav_id, solution.paths[uav_id]


def _pick_two_paths(solution: Solution) -> tuple[int, int, list[int], list[int]]:
    """Wybierz dwie różne losowe ścieżki."""
    uav1 = random.randint(0, solution.num_uavs - 1)
    uav2 = random.choice([i for i in range(solution.num_uavs) if i != uav1])
    return uav1, uav2, solution.paths[uav1], solution.paths[uav2]


# ── Mutacje intra-route ───────────────────────────────────────


def swap_mutation(solution: Solution) -> Solution:
    """Zamień dwa losowe regiony w tej samej ścieżce."""
    sol = solution.copy()
    _, path = _pick_path(sol)
    if len(path) < 2:
        return sol
    pos1, pos2 = random.sample(range(len(path)), 2)
    path[pos1], path[pos2] = path[pos2], path[pos1]
    return sol


def invert_mutation(solution: Solution) -> Solution:
    """Odwróć losowy segment ścieżki (2-opt intra-route)."""
    sol = solution.copy()
    _, path = _pick_path(sol)
    if len(path) < 2:
        return sol
    cut1, cut2 = sorted(random.sample(range(len(path) + 1), 2))
    path[cut1:cut2] = list(reversed(path[cut1:cut2]))
    return sol


def or_opt_mutation(solution: Solution) -> Solution:
    """
    Or-opt: przenieś spójny blok k regionów (k=2 lub 3) do innego miejsca.

    Dwa warianty z równym prawdopodobieństwem:
      - intra-route: wstaw blok w inne miejsce TEJ SAMEJ ścieżki
      - inter-route: przenieś blok do INNEJ ścieżki drona

    Or-opt uogólnia 2-opt przez przenoszenie bloków zamiast pojedynczych
    regionów — skuteczny dla VRP gdzie sąsiadujące regiony często powinny
    pozostać razem (Or 1976, Prins 2004).

    k=1 odpowiada relokacji pojedynczego regionu (insert/relocate),
    k=2 i k=3 przenoszą spójne bloki. Or (1976) definiuje Or-opt
    dla k ∈ {1, 2, 3}.
    """
    sol = solution.copy()
    k = random.choice([1, 2, 3])

    if random.random() < 0.5:
        # --- intra-route ---
        _, path = _pick_path(sol)
        if len(path) <= k:
            return sol
        start = random.randint(0, len(path) - k)
        block = path[start:start + k]
        del path[start:start + k]
        insert_pos = random.randint(0, len(path))
        path[insert_pos:insert_pos] = block
    else:
        # --- inter-route ---
        if solution.num_uavs < 2:
            return sol
        uav1, uav2, src, dst = _pick_two_paths(sol)
        if len(src) <= k:
            return sol
        start = random.randint(0, len(src) - k)
        block = src[start:start + k]
        del src[start:start + k]
        insert_pos = random.randint(0, len(dst))
        dst[insert_pos:insert_pos] = block

    return sol


# ── Mutacje inter-route ───────────────────────────────────────


def transfer_mutation(solution: Solution) -> Solution:
    """Przenieś jeden region z jednej ścieżki do drugiej."""
    if solution.num_uavs < 2:
        return solution
    sol = solution.copy()
    _, _, src, dst = _pick_two_paths(sol)
    if not src:
        return sol
    element = src.pop(random.randint(0, len(src) - 1))
    dst.insert(random.randint(0, len(dst)), element)
    return sol


def exchange_mutation(solution: Solution) -> Solution:
    """Zamień jeden region między dwiema ścieżkami (1:1 swap inter-route)."""
    if solution.num_uavs < 2:
        return solution
    sol = solution.copy()
    _, _, path1, path2 = _pick_two_paths(sol)
    if not path1 or not path2:
        return sol
    pos1 = random.randint(0, len(path1) - 1)
    pos2 = random.randint(0, len(path2) - 1)
    path1[pos1], path2[pos2] = path2[pos2], path1[pos1]
    return sol


def cross_route_mutation(solution: Solution) -> Solution:
    """
    2-opt* (cross-route): zamień ogony dwóch ścieżek za losowym punktem podziału.

    Dron1: [A B | C D E]  →  Dron1: [A B | F G]
    Dron2: [X Y | F G]    →  Dron2: [X Y | C D E]

    To pozwala na dużą restrukturyzację podziału regionów między dronami
    — transfer i exchange zmieniają po jednym regionie, cross_route wymienia
    całe segmenty końcowe. Kluczowy operator dla równoważenia obciążenia
    (Potvin & Rousseau 1995).
    """
    if solution.num_uavs < 2:
        return solution
    sol = solution.copy()
    _, _, path1, path2 = _pick_two_paths(sol)

    # Punkt podziału musi zostawić co najmniej 1 element w każdej części
    if len(path1) < 2 or len(path2) < 2:
        return sol

    cut1 = random.randint(1, len(path1) - 1)
    cut2 = random.randint(1, len(path2) - 1)

    tail1 = path1[cut1:]
    tail2 = path2[cut2:]

    path1[cut1:] = tail2
    path2[cut2:] = tail1

    return sol


# ── Dispatcher ───────────────────────────────────────────────

_OPERATORS = [
    invert_mutation,
    or_opt_mutation,
    transfer_mutation,
    swap_mutation,
    cross_route_mutation,
    exchange_mutation,
]

# Domyślne wagi — uniform (brak wiedzy a priori o operatorach)
_DEFAULT_WEIGHTS = [1/6, 1/6, 1/6, 1/6, 1/6, 1/6]


def mutate(
    solution: Solution,
    test_case: TestCase,
    mutation_rate: float = 0.1,
    weights: list[float] | None = None,
) -> Solution:
    """
    Zastosuj losową mutację z podanymi wagami operatorów.

    weights: opcjonalna lista 6 wag (invert, or_opt, transfer, swap,
             cross_route, exchange). None = domyślne wagi.
             Nie muszą sumować się do 1 (random.choices normalizuje).
    """
    if random.random() > mutation_rate:
        return solution

    w = weights if weights is not None else _DEFAULT_WEIGHTS
    if sum(w) == 0:
        w = _DEFAULT_WEIGHTS
    mutated = random.choices(_OPERATORS, weights=w)[0](solution)

    if not mutated.validate(test_case):
        return solution

    return mutated