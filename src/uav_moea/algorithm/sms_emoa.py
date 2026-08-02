"""
SMS-EMOA: S-Metric Selection Evolutionary Multi-Objective Algorithm.

Główna pętla algorytmu:
1. Inicjalizuj populację (hybrydowo)
2. Powtarzaj:
   a. Wybierz dwóch rodziców
   b. Crossover (OX)
   c. Mutacja
   d. Ewaluacja potomka
   e. Dodaj potomka do populacji (μ → μ+1)
   f. Usuń najgorszego po hypervolume contribution (μ+1 → μ)
3. Zwróć archiwum (front Pareto)
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
from uav_moea.algorithm.initializers import initialize_population, generate_balanced_random

logger = logging.getLogger(__name__)


# ── Dominacja i sortowanie ───────────────────────────────────────


def fast_non_dominated_sort(population: list[Solution]) -> list[list[int]]:
    """
    Szybkie sortowanie niedominowane

    Zwraca listę frontów — każdy front to lista indeksów do population.
    Front 0 = rozwiązania niedominowane (Pareto front).
    """
    n = len(population)
    f1 = [s.f1 for s in population]
    f2 = [s.f2 for s in population]

    domination_count = [0] * n
    dominates = [[] for _ in range(n)]

    for i in range(n):
        for j in range(i + 1, n):
            # i dominuje j?
            if f1[i] <= f1[j] and f2[i] <= f2[j] and (f1[i] < f1[j] or f2[i] < f2[j]):
                dominates[i].append(j)
                domination_count[j] += 1
            # j dominuje i?
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


# ── Hypervolume ──────────────────────────────────────────────────


def hypervolume_2d(points: list[tuple[float, float]], ref: tuple[float, float]) -> float:
    """
    Hypervolume 2D.

    Sortuje punkty po f1, liczy pole prostokątów między kolejnymi
    punktami a punktem referencyjnym.
    """
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
class SmsEmoaConfig:
    """Parametry SMS-EMOA."""

    population_size: int = 100
    max_evaluations: int = 10_000
    crossover_prob: float = 0.9
    mutation_rate: float = 0.1
    num_uavs: int = 3
    seed: int | None = None
    snapshot_interval: int = 100
    mutation_weights: list[float] | None = None
    init_strategy: str = "hybrid"  # "hybrid" | "random"


# ── Algorytm ─────────────────────────────────────────────────────


class SmsEmoa:
    """SMS-EMOA optimizer."""

    def __init__(self, config: SmsEmoaConfig, test_case: TestCase):
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
        """Wygeneruj i ewaluuj populację startową."""
        if self.config.init_strategy == "random":
            self.population = [
                generate_balanced_random(self.test_case, self.config.num_uavs)
                for _ in range(self.config.population_size)
            ]
        else:
            self.population = initialize_population(
                self.test_case,
                self.config.num_uavs,
                self.config.population_size,
            )

        for sol in self.population:
            evaluate(sol, self.test_case)
            self.evaluations += 1

        # Punkt referencyjny: najgorsze wartości + 10% margines
        worst_f1 = max(sol.f1 for sol in self.population)
        worst_f2 = max(sol.f2 for sol in self.population)
        self.ref_point = (worst_f1 * 1.1, worst_f2 * 1.1)

        logger.info(
            "Population initialized: %d solutions, ref_point=(%.2f, %.2f)",
            len(self.population), self.ref_point[0], self.ref_point[1],
        )

    # ── Selekcja do usunięcia ────────────────────────────────────

    def _normalize(self, f1: float, f2: float) -> tuple[float, float]:
        """Normalizuj punkt do [0,1] na podstawie całej populacji (μ+1)."""
        all_f1 = [s.f1 for s in self.population]
        all_f2 = [s.f2 for s in self.population]
        min_f1, max_f1 = min(all_f1), max(all_f1)
        min_f2, max_f2 = min(all_f2), max(all_f2)
        range_f1 = max_f1 - min_f1 if max_f1 != min_f1 else 1.0
        range_f2 = max_f2 - min_f2 if max_f2 != min_f2 else 1.0
        return ((f1 - min_f1) / range_f1, (f2 - min_f2) / range_f2)

    def _select_for_removal(self) -> tuple[int, list[list[int]]]:
        fronts = fast_non_dominated_sort(self.population)
        worst_front = fronts[-1]

        if len(worst_front) == 1:
            return worst_front[0], fronts

        # 1. Wyciągnij i znormalizuj punkty z najgorszego frontu
        #    Normalizacja względem całej populacji μ+1 (zgodnie z pymoo)
        points = []
        for i in worst_front:
            sol = self.population[i]
            points.append(self._normalize(sol.f1, sol.f2))

        # 2. Posortuj po f1, zapamiętaj oryginalne pozycje
        order = sorted(range(len(points)), key=lambda k: points[k][0])

        # 3. Policz wkład HV każdego punktu
        ref = (1.1, 1.1)
        contributions = [0.0] * len(points)

        for pos, idx in enumerate(order):
            f1, f2 = points[idx]

            # Szerokość: do sąsiada po prawej, lub do ref
            if pos < len(order) - 1:
                right_f1 = points[order[pos + 1]][0]
            else:
                right_f1 = ref[0]
            width = right_f1 - f1

            # Wysokość: do sąsiada po lewej, lub do ref
            if pos > 0:
                left_f2 = points[order[pos - 1]][1]
            else:
                left_f2 = ref[1]
            height = left_f2 - f2

            contributions[idx] = width * height

        # 4. Usuń punkt z najmniejszym wkładem
        worst_idx = contributions.index(min(contributions))
        return worst_front[worst_idx], fronts

    # ── Archiwum ─────────────────────────────────────────────────

    def _update_archive(self) -> None:
        """Aktualizuj archiwum — zachowaj tylko niedominowane rozwiązania."""
        fronts = fast_non_dominated_sort(self.population)
        if not fronts:
            return

        front0 = [self.population[i] for i in fronts[0]]
        combined = self.archive + front0

        # Deduplikacja po (f1, f2)
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
        """Zapisz statystyki aktualnej generacji."""
        fronts = fast_non_dominated_sort(self.population)
        front0 = fronts[0] if fronts else []

        # Hypervolume frontu Pareto
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
        """Uruchom SMS-EMOA. Zwraca archiwum (front Pareto)."""
        logger.info(
            "SMS-EMOA: pop=%d, max_eval=%d, UAVs=%d, test_case=%s (%d regions)",
            self.config.population_size,
            self.config.max_evaluations,
            self.config.num_uavs,
            self.test_case.name,
            self.test_case.node_count,
        )

        start_time = time.time()

        # Inicjalizacja
        self._init_population()
        self._update_archive()
        self._record_stats(save_snapshot=True)

        # Pętla ewolucyjna
        while self.evaluations < self.config.max_evaluations:
            self.generation += 1

            # Selekcja rodziców
            parent1, parent2 = random.sample(self.population, 2)

            # Crossover
            if random.random() < self.config.crossover_prob:
                offspring = crossover(parent1, parent2, self.test_case)
            else:
                offspring = parent1.copy()

            # Mutacja
            offspring = mutate(
                offspring,
                self.test_case,
                self.config.mutation_rate,
                weights=self.config.mutation_weights,
            )       

            # Ewaluacja
            evaluate(offspring, self.test_case)
            self.evaluations += 1

            # Dodaj potomka (μ → μ+1)
            self.population.append(offspring)

            # Usuń najgorszego (μ+1 → μ)
            remove_idx, _ = self._select_for_removal()
            self.population[remove_idx] = self.population[-1]
            self.population.pop()

            # Aktualizuj archiwum co 10 generacji
            if self.generation % 10 == 0:
                self._update_archive()

            # Loguj co 100 generacji
            if self.generation % 100 == 0:
                save_snapshot = self.generation % self.config.snapshot_interval == 0
                self._record_stats(save_snapshot=save_snapshot)

                elapsed = time.time() - start_time
                hv = self.history["hypervolume"][-1]
                logger.info(
                    "Gen %5d | Eval %6d | HV: %10.2f | f1: %8.2f | f2: %6.4f | %.1fs",
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
        """Zapisz wyniki do JSON."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        results = {
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