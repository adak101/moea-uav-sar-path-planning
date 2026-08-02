"""
Generacyjny SMS-EMOA — wariant ablacyjny (czwarty narożnik tabeli 2x2).

  | architektura \\ selekcja | HV-contribution | crowding distance |
  |--------------------------|-----------------|-------------------|
  | steady-state             | SMS-EMOA        | SS-NSGA-II        |
  | generacyjny              | GEN-SMS-EMOA    | NSGA-II           |   <- ten plik

Architektura IDENTYCZNA jak NSGA-II (generacyjny, μ+μ):
1. Inicjalizuj populację P (μ)
2. Powtarzaj (jedna generacja = μ ewaluacji):
   a. Stwórz μ potomków przez losowy wybór rodziców + crossover + mutację
   b. Połącz R = P ∪ Q (2μ)
   c. Posortuj na fronty niezdominowane
   d. Wypełnij P frontami od najlepszego; ostatni front docinaj
      ITERACYJNIE usuwając osobnika o najmniejszym wkładzie HV
      (z przeliczeniem wkładów po każdym usunięciu)

Jedyna różnica względem NSGA-II: krok (d) — docinanie ostatniego
frontu przez wkład do hyperwolumenu zamiast crowding distance.

Uwaga implementacyjna: wkład do HV zależy od sąsiedztwa na froncie,
więc po każdym usunięciu trzeba przeliczyć wkłady pozostałych — stąd
docinanie iteracyjne (a nie jednoprzebiegowe jak w NSGA-II z CD).
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


# ── Dominacja i sortowanie (identyczne jak NSGA-II) ──────────────


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


# ── Crowding distance (do selekcji środowiskowej) ────────────────


def crowding_distance(front_indices: list[int], population: list[Solution]) -> dict[int, float]:
    """Crowding distance dla jednego frontu (jak w NSGA-II)."""
    if len(front_indices) <= 2:
        return {i: float('inf') for i in front_indices}

    distances = {i: 0.0 for i in front_indices}

    for obj_attr in ('f1', 'f2'):
        sorted_front = sorted(front_indices, key=lambda i: getattr(population[i], obj_attr))

        obj_min = getattr(population[sorted_front[0]], obj_attr)
        obj_max = getattr(population[sorted_front[-1]], obj_attr)
        obj_range = obj_max - obj_min if obj_max != obj_min else 1.0

        distances[sorted_front[0]] = float('inf')
        distances[sorted_front[-1]] = float('inf')

        for k in range(1, len(sorted_front) - 1):
            prev_val = getattr(population[sorted_front[k - 1]], obj_attr)
            next_val = getattr(population[sorted_front[k + 1]], obj_attr)
            distances[sorted_front[k]] += (next_val - prev_val) / obj_range

    return distances


# ── Hypervolume ──────────────────────────────────────────────────


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


# ── Wkład do HV (operuje na podanej liście rozwiązań) ────────────


def _normalize_points(front_sols: list[Solution], all_sols: list[Solution]) -> list[tuple[float, float]]:
    """Normalizacja front_sols używając granic z all_sols (cała populacja).

    Spójna z SMS-EMOA: normalizacja względem całej populacji μ+1,
    nie tylko ostatniego frontu (Beume et al. 2007, pymoo).
    """
    f1s = [s.f1 for s in all_sols]
    f2s = [s.f2 for s in all_sols]
    min_f1, max_f1 = min(f1s), max(f1s)
    min_f2, max_f2 = min(f2s), max(f2s)
    r1 = max_f1 - min_f1 if max_f1 != min_f1 else 1.0
    r2 = max_f2 - min_f2 if max_f2 != min_f2 else 1.0
    return [((s.f1 - min_f1) / r1, (s.f2 - min_f2) / r2) for s in front_sols]


def _hv_contributions(norm_pts: list[tuple[float, float]]) -> list[float]:
    """
    Wkład każdego punktu do HV frontu (2D, znormalizowane, ref=(1.1,1.1)).

    Zwraca listę wkładów w tej samej kolejności co norm_pts.
    Wkład i-tego punktu = (f1 prawego sąsiada - f1 tego)
                        * (f2 lewego sąsiada - f2 tego).
    """
    ref = (1.1, 1.1)
    n = len(norm_pts)
    # Posortuj pozycje po f1 rosnąco
    order = sorted(range(n), key=lambda k: norm_pts[k][0])

    contrib = [0.0] * n
    for pos, k in enumerate(order):
        f1, f2 = norm_pts[k]
        if pos < len(order) - 1:
            right_f1 = norm_pts[order[pos + 1]][0]
        else:
            right_f1 = ref[0]
        if pos > 0:
            left_f2 = norm_pts[order[pos - 1]][1]
        else:
            left_f2 = ref[1]
        width = right_f1 - f1
        height = left_f2 - f2
        contrib[k] = width * height
    return contrib


def _truncate_front_by_hv(
    front_sols: list[Solution],
    n_keep: int,
    other_sols: list[Solution],
) -> list[int]:
    """
    Dotnij front_sols do n_keep, iteracyjnie usuwając osobnika
    o najmniejszym wkładzie HV (z przeliczeniem po każdym usunięciu).

    other_sols: rozwiązania z lepszych frontów (wszystkie przetrwają).
    Normalizacja używa other_sols + bieżących ocalałych z frontu —
    spójna z SMS-EMOA gdzie normalizacja obejmuje całą populację μ+1.
    """
    survivors = list(range(len(front_sols)))

    while len(survivors) > n_keep:
        current_front = [front_sols[p] for p in survivors]
        all_current = other_sols + current_front
        norm = _normalize_points(current_front, all_current)
        contrib = _hv_contributions(norm)
        worst_local = min(range(len(survivors)), key=lambda k: contrib[k])
        survivors.pop(worst_local)

    return survivors


# ── Konfiguracja ─────────────────────────────────────────────────


@dataclass
class GenSmsEmoaConfig:
    """Parametry generacyjnego SMS-EMOA."""
    population_size: int = 100
    max_evaluations: int = 10_000
    crossover_prob: float = 0.9
    mutation_rate: float = 0.1
    num_uavs: int = 3
    seed: int | None = None
    snapshot_interval: int = 100
    mutation_weights: list[float] | None = None


# ── Algorytm ─────────────────────────────────────────────────────


class GenSmsEmoa:
    """Generacyjny SMS-EMOA optimizer."""

    def __init__(self, config: GenSmsEmoaConfig, test_case: TestCase):
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

    # ── Inicjalizacja (identyczna jak NSGA-II) ───────────────────

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

    # ── Selekcja środowiskowa (TU JEST RÓŻNICA vs NSGA-II) ───────

    def _environmental_selection(self, combined: list[Solution]) -> list[Solution]:
        """
        Z 2μ wybierz μ: pełne fronty od najlepszego, ostatni front
        docięty przez iteracyjne HV-contribution (zamiast crowding distance).
        """
        mu = self.config.population_size
        fronts = fast_non_dominated_sort(combined)

        new_population: list[Solution] = []
        for front in fronts:
            if len(new_population) + len(front) <= mu:
                new_population.extend(combined[i] for i in front)
            else:
                slots = mu - len(new_population)
                front_sols = [combined[i] for i in front]
                kept_pos = _truncate_front_by_hv(front_sols, slots, new_population)
                new_population.extend(front_sols[p] for p in kept_pos)
                break

        return new_population

    # ── Archiwum (identyczne jak NSGA-II) ────────────────────────

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

    # ── Statystyki (identyczne jak NSGA-II) ──────────────────────

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

    # ── Główna pętla (generacyjna, identyczna jak NSGA-II) ───────

    def run(self) -> list[Solution]:
        logger.info(
            "GEN-SMS-EMOA: pop=%d, max_eval=%d, UAVs=%d, test_case=%s (%d regions)",
            self.config.population_size,
            self.config.max_evaluations,
            self.config.num_uavs,
            self.test_case.name,
            self.test_case.node_count,
        )

        start_time = time.time()
        mu = self.config.population_size

        self._init_population()
        self._update_archive()
        self._record_stats(save_snapshot=True)

        while self.evaluations < self.config.max_evaluations:
            self.generation += 1

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

            combined = self.population + offspring_pop
            self.population = self._environmental_selection(combined)

            if self.generation % 10 == 0:
                self._update_archive()

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

        self._update_archive()
        self._record_stats(save_snapshot=True)

        elapsed = time.time() - start_time
        logger.info(
            "Done: %d generations, %d evaluations, archive=%d, HV=%.2f, %.1fs",
            self.generation, self.evaluations, len(self.archive),
            self.history["hypervolume"][-1], elapsed,
        )
        return self.archive

    # ── Zapis wyników (inna etykieta) ────────────────────────────

    def save_results(self, filepath: str | Path) -> None:
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        results = {
            "algorithm": "GEN-SMS-EMOA",
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