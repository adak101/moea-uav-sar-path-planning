"""
NSGA-II: Non-dominated Sorting Genetic Algorithm II (Deb et al., 2002).

Klasyczna, generacyjna wersja:
1. Inicjalizuj populację P rozmiaru μ
2. Powtarzaj (jedna "generacja" = μ ewaluacji):
   a. Stwórz potomstwo Q rozmiaru μ przez:
      - losowy wybór rodziców (random.sample)
      - crossover (OX)
      - mutację
   b. Połącz R = P ∪ Q (rozmiar 2μ)
   c. Posortuj R na fronty niezdominowane
   d. Wypełnij nową populację P frontami od najlepszego,
      ostatni front docinając wg crowding distance (descending)
"""
from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path

from uav_moea.io.data_loader import TestCase
from uav_moea.model.individual import Solution
from uav_moea.model.objectives import evaluate
from uav_moea.algorithm.operators import crossover, mutate
from uav_moea.algorithm.initializers import initialize_population

logger = logging.getLogger(__name__)


# ── Dominacja i sortowanie (jak w SMS-EMOA, identyczne) ──────────


def fast_non_dominated_sort(population: list[Solution]) -> list[list[int]]:
    """Szybkie sortowanie niezdominowane. Zwraca listę frontów (indeksy)."""
    n = len(population)
    f1 = [s.f1 for s in population]
    f2 = [s.f2 for s in population]

    domination_count = [0] * n
    dominates = [[] for _ in range(n)]

    for i in range(n):
        for j in range(i + 1, n):
            if f1[i] <= f1[j] and f2[i] <= f2[j] and (f1[i] < f1[j] or f2[i] < f2[j]):
                dominates[i].append(j)
                domination_count[j] += 1
            elif f1[j] <= f1[i] and f2[j] <= f2[i] and (f1[j] < f1[i] or f2[j] < f2[i]):
                dominates[j].append(i)
                domination_count[i] += 1

    fronts = []
    current = [i for i in range(n) if domination_count[i] == 0]
    while current:
        fronts.append(current)
        next_front = []
        for i in current:
            for j in dominates[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    next_front.append(j)
        current = next_front
    return fronts


# ── Crowding distance ────────────────────────────────────────────


def crowding_distance(front_indices: list[int], population: list[Solution]) -> dict[int, float]:
    """
    Crowding distance dla jednego frontu.

    Punkty brzegowe dostają inf (zawsze zachowywane).
    Punkty w środku: suma znormalizowanych odległości do sąsiadów w każdym celu.
    """
    if len(front_indices) <= 2:
        # Wszystkie brzegowe
        return {i: float('inf') for i in front_indices}

    distances = {i: 0.0 for i in front_indices}

    for obj_attr in ('f1', 'f2'):
        # Sortuj po danym celu
        sorted_front = sorted(front_indices, key=lambda i: getattr(population[i], obj_attr))

        obj_min = getattr(population[sorted_front[0]], obj_attr)
        obj_max = getattr(population[sorted_front[-1]], obj_attr)
        obj_range = obj_max - obj_min if obj_max != obj_min else 1.0

        # Brzegowe = inf
        distances[sorted_front[0]] = float('inf')
        distances[sorted_front[-1]] = float('inf')

        # Środkowe
        for k in range(1, len(sorted_front) - 1):
            prev_val = getattr(population[sorted_front[k - 1]], obj_attr)
            next_val = getattr(population[sorted_front[k + 1]], obj_attr)
            distances[sorted_front[k]] += (next_val - prev_val) / obj_range

    return distances


# ── Hypervolume (do raportowania, identyczne jak w SMS-EMOA) ─────


def hypervolume_2d(points: list[tuple[float, float]], ref: tuple[float, float]) -> float:
    if not points:
        return 0.0
    sorted_pts = sorted(points, key=lambda p: (p[0], -p[1]))
    hv = 0.0
    prev_f2 = ref[1]
    for f1, f2 in sorted_pts:
        width = ref[0] - f1
        height = prev_f2 - f2
        if width > 0 and height > 0:
            hv += width * height
        prev_f2 = f2
    return hv


# ── Konfiguracja ─────────────────────────────────────────────────


@dataclass
class Nsga2Config:
    """Parametry NSGA-II."""
    population_size: int = 100
    max_evaluations: int = 10_000
    crossover_prob: float = 0.9
    mutation_rate: float = 0.1
    num_uavs: int = 3
    seed: int | None = None
    snapshot_interval: int = 100
    mutation_weights: list[float] | None = None


# ── Algorytm ─────────────────────────────────────────────────────


class Nsga2:
    """NSGA-II optimizer (klasyczny, generacyjny)."""

    def __init__(self, config: Nsga2Config, test_case: TestCase):
        self.config = config
        self.test_case = test_case

        if config.seed is not None:
            random.seed(config.seed)

        self.population: list[Solution] = []
        self.archive: list[Solution] = []
        self.evaluations = 0
        self.generation = 0
        self.ref_point: tuple[float, float] | None = None

        self.history: dict = {
            "generations": [],
            "evaluations": [],
            "hypervolume": [],
            "best_f1": [],
            "best_f2": [],
            "snapshots": [],
        }

    # ── Inicjalizacja ────────────────────────────────────────────

    def _init_population(self) -> None:
        self.population = initialize_population(
            self.test_case,
            self.config.num_uavs,
            self.config.population_size,
        )
        for sol in self.population:
            evaluate(sol, self.test_case)
            self.evaluations += 1

        worst_f1 = max(sol.f1 for sol in self.population)
        worst_f2 = max(sol.f2 for sol in self.population)
        self.ref_point = (worst_f1 * 1.1, worst_f2 * 1.1)

        logger.info(
            "Population initialized: %d solutions, ref_point=(%.2f, %.2f)",
            len(self.population), self.ref_point[0], self.ref_point[1],
        )

    # ── Selekcja środowiskowa (rank → crowding distance) ─────────

    def _environmental_selection(self, combined: list[Solution]) -> list[Solution]:
        """Z populacji 2μ wybierz μ najlepszych po: rank, potem crowding distance."""
        mu = self.config.population_size
        fronts = fast_non_dominated_sort(combined)

        new_population = []
        for front in fronts:
            if len(new_population) + len(front) <= mu:
                # Cały front się zmieści
                new_population.extend([combined[i] for i in front])
            else:
                # Trzeba dociąć front: weź te z największym crowding distance
                distances = crowding_distance(front, combined)
                sorted_front = sorted(front, key=lambda i: -distances[i])
                slots = mu - len(new_population)
                new_population.extend([combined[i] for i in sorted_front[:slots]])
                break

        return new_population

    # ── Archiwum ─────────────────────────────────────────────────

    def _update_archive(self) -> None:
        fronts = fast_non_dominated_sort(self.population)
        if not fronts:
            return

        front0 = [self.population[i] for i in fronts[0]]
        combined = self.archive + front0

        unique = {}
        for sol in combined:
            key = (round(sol.f1, 6), round(sol.f2, 6))
            if key not in unique:
                unique[key] = sol

        combined = list(unique.values())
        combined_fronts = fast_non_dominated_sort(combined)
        self.archive = [combined[i] for i in combined_fronts[0]]

    # ── Statystyki ───────────────────────────────────────────────

    def _record_stats(self, save_snapshot: bool = False) -> None:
        fronts = fast_non_dominated_sort(self.population)
        front0 = fronts[0] if fronts else []

        if front0 and self.ref_point:
            points = [(self.population[i].f1, self.population[i].f2) for i in front0]
            hv = hypervolume_2d(points, self.ref_point)
        else:
            hv = 0.0

        best_f1 = min(sol.f1 for sol in self.population)
        best_f2 = min(sol.f2 for sol in self.population)

        self.history["generations"].append(self.generation)
        self.history["evaluations"].append(self.evaluations)
        self.history["hypervolume"].append(hv)
        self.history["best_f1"].append(best_f1)
        self.history["best_f2"].append(best_f2)

        if save_snapshot:
            self.history["snapshots"].append({
                "generation": self.generation,
                "population": [{"f1": s.f1, "f2": s.f2} for s in self.population],
            })

    # ── Główna pętla ─────────────────────────────────────────────

    def run(self) -> list[Solution]:
        """Uruchom NSGA-II. Zwraca archiwum (front Pareto)."""
        logger.info(
            "NSGA-II: pop=%d, max_eval=%d, UAVs=%d, test_case=%s (%d regions)",
            self.config.population_size,
            self.config.max_evaluations,
            self.config.num_uavs,
            self.test_case.name,
            self.test_case.node_count,
        )

        start_time = time.time()
        mu = self.config.population_size

        # Inicjalizacja
        self._init_population()
        self._update_archive()
        self._record_stats(save_snapshot=True)

        # Pętla generacyjna
        while self.evaluations < self.config.max_evaluations:
            self.generation += 1

            # Stwórz potomstwo Q rozmiaru μ
            offspring_pop = []
            for _ in range(mu):
                if self.evaluations >= self.config.max_evaluations:
                    break

                parent1, parent2 = random.sample(self.population, 2)

                if random.random() < self.config.crossover_prob:
                    child = crossover(parent1, parent2, self.test_case)
                else:
                    child = parent1.copy()

                child = mutate(child, self.test_case, self.config.mutation_rate,
                              weights=self.config.mutation_weights)
                evaluate(child, self.test_case)
                self.evaluations += 1
                offspring_pop.append(child)

            # 3. Selekcja środowiskowa: 2μ → μ
            combined = self.population + offspring_pop
            self.population = self._environmental_selection(combined)

            # Aktualizuj archiwum co 10 generacji
            if self.generation % 10 == 0:
                self._update_archive()

            # Loguj i zapisz statystyki co generację (NSGA-II ma mało generacji vs SMS-EMOA)
            save_snapshot = self.generation % max(1, self.config.snapshot_interval // mu) == 0
            self._record_stats(save_snapshot=save_snapshot)

            if self.generation % 10 == 0:
                elapsed = time.time() - start_time
                hv = self.history["hypervolume"][-1]
                logger.info(
                    "Gen %4d | Eval %6d | HV: %10.2f | f1: %8.2f | f2: %6.4f | %.1fs",
                    self.generation, self.evaluations, hv,
                    self.history["best_f1"][-1],
                    self.history["best_f2"][-1],
                    elapsed,
                )

        # Finalizacja
        self._update_archive()
        self._record_stats(save_snapshot=True)

        elapsed = time.time() - start_time
        logger.info(
            "Done: %d generations, %d evaluations, archive=%d, HV=%.2f, %.1fs",
            self.generation, self.evaluations, len(self.archive),
            self.history["hypervolume"][-1], elapsed,
        )
        return self.archive

    # ── Zapis wyników ────────────────────────────────────────────

    def save_results(self, filepath: str | Path) -> None:
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        results = {
            "algorithm": "NSGA-II",
            "config": {
                "population_size": self.config.population_size,
                "max_evaluations": self.config.max_evaluations,
                "num_uavs": self.config.num_uavs,
                "test_case": self.test_case.name,
                "seed": self.config.seed,
            },
            "history": self.history,
            "archive": [
                {"paths": sol.paths, "f1": sol.f1, "f2": sol.f2}
                for sol in self.archive
            ],
        }

        with open(filepath, "w") as f:
            json.dump(results, f, indent=2)
        logger.info("Results saved to %s", filepath)