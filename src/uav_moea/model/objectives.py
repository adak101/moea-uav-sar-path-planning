"""
Funkcje celu dla optymalizacji ścieżek UAV.

f1: Czas zakończenia misji (max czas ze wszystkich dronów)
f2: Znormalizowany współczynnik nieodkrytego obszaru
    f2 = 1 - ∫V(t)dt / (V_total * T_cut)
"""

from __future__ import annotations

from dataclasses import dataclass

from uav_moea.io.data_loader import TestCase
from uav_moea.model.individual import Solution


# Harmonogram lotu


@dataclass
class ScheduleEntry:
    """Pojedynczy punkt w harmonogramie lotu drona"""

    region: int
    arrival_time: float
    departure_time: float
    victims: int


def build_schedule(path: list[int], test_case: TestCase) -> list[ScheduleEntry]:
    """Zbuduj harmonogram lotu dla danego drona"""

    schedule = []
    current_time = 0.0

    for i, region in enumerate(path):
        if i == 0:
            arrival = test_case.get_reach_time(region)
        else:
            arrival = current_time + test_case.get_flight_time(path[i-1], region)
        
        departure = arrival + test_case.get_scan_time(region)
        victims = test_case.get_population(region)

        schedule.append(ScheduleEntry(region, arrival, departure, victims))
        current_time = departure
    
    return schedule

def build_all_schedules(solution: Solution, test_case: TestCase) -> list[list[ScheduleEntry]]:
    """Zbuduj harmonogramy dla wszystkich dronów"""

    return [build_schedule(path, test_case) for path in solution.paths]


def compute_f1(schedules: list[list[ScheduleEntry]]) -> float:
    """f1 = max departure time ze wszystkich dronów (najwolniejszy dron wyznacza czas zakonczenia misji)

    Dron z pustą trasą (schedule == []) nie wnosi wkładu do maksimum
    (przyjmuje się dla niego czas zakończenia 0).
    """

    max_time = 0.0

    for schedule in schedules:
        if not schedule:
            continue
        max_time = max(max_time, schedule[-1].departure_time)

    return max_time

def compute_f2(schedules: list[list[ScheduleEntry]], total_victims) -> float:
    """
    f2 = 1 - ∫₀ᵀᶜᵘᵗ V(t)dt / (V_total * T_cut)

    Model dynamiczny (liniowy) - zgodny z [Trojanowski et al. 2023]:
    Podczas skanowania regionu liczba odnalezionych ofiar rośnie LINIOWO
    od zera do ENV regionu w czasie scan_time, nie skokowo na końcu.

    T_cut = czas zakończenia operacji najszybszego drona (decyzja celowa:
    eliminuje degeneracyjne rozwiązania w których jeden dron jest sztucznie
    obciążany pojedynczym regionem na końcu, zawyżając f2).

    Wkład regionu R do całki (arrival=a, departure=d, victims=V):
      - Skan ZAKOŃCZONY przed T_cut (d <= T_cut):
          trójkąt liniowego wzrostu [a,d]: V * scan_time / 2
          prostokąt po skanie [d, T_cut]: V * (T_cut - d)
          razem: V * (T_cut - (a+d)/2)

      - Skan TRWA w T_cut (a < T_cut < d):
          częściowa całka liniowego wzrostu od a do T_cut:
          V * (T_cut - a)² / (2 * scan_time)

      - Skan nie rozpoczęty (a >= T_cut): brak wkładu
    """
    completion_times = [s[-1].departure_time for s in schedules if s]
    t_cut = min(completion_times)

    # Teoretyczny przypadek: wszystko znalezione od razu
    if t_cut <= 0:
        return 1.0
    # Nikt nie został odnaleziony
    if total_victims <= 0:
        return 0.0

    integral = 0.0
    for schedule in schedules:
        for entry in schedule:
            if entry.victims <= 0:
                continue
            if entry.arrival_time >= t_cut:
                # Skan jeszcze nie rozpoczął się przed T_cut
                continue

            scan_time = entry.departure_time - entry.arrival_time

            if entry.departure_time <= t_cut:
                # Skan zakończony przed T_cut - pełny wkład (trójkąt + prostokąt)
                midpoint = (entry.arrival_time + entry.departure_time) / 2
                integral += entry.victims * (t_cut - midpoint)
            else:
                # Skan trwa w T_cut - częściowy wkład (tylko trójkąt do T_cut)
                if scan_time > 0:
                    partial = entry.victims * (t_cut - entry.arrival_time) ** 2 / (2 * scan_time)
                    integral += partial

    f2 = 1.0 - integral / (total_victims * t_cut)
    return f2


# ----Ewaluacja rozwiazania----------------


def evaluate(solution: Solution, test_case: TestCase) -> tuple[float, float]:
    """
    Stworz harmonogramy i oblicz funkcje celu

    Wynik zapisywany w solution.f1 i solution.f2

    """

    schedules = build_all_schedules(solution, test_case)

    f1 = compute_f1(schedules)
    f2 = compute_f2(schedules, test_case.total_population)

    solution.f1 = f1
    solution.f2 = f2


    return f1, f2
    
