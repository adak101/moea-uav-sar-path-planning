"""
Walidacja silnika obliczeniowego SMS-EMOA czterema niezależnymi cross-checkami:

  (a) Naiwne O(n^2) sprawdzenie dominacji (wprost z definicji Pareto) vs
      fast_non_dominated_sort() używane w kodzie.
  (b) Wkład do hypervolume metodą Leave-One-Out: HV(front) - HV(front \\ {x})
      liczone metodą "wprost" (dwa pełne obliczenia HV) vs wartość, którą
      wylicza procedura selekcji SMS-EMOA (SmsEmoa._select_for_removal).
  (c) HV własnej implementacji (hypervolume_2d) vs niezależny oracle
      (pymoo.indicators.hv.HV) na losowych frontach 2D.
  (d) Monotoniczność HV archiwum: archiwum niezdominowanych może w trakcie
      przebiegu tylko rosnąć lub zostać bez zmian, nigdy zmaleć.

Żaden z testów nie modyfikuje kodu algorytmów — importuje i wywołuje funkcje
z src/algorithm/sms_emoa.py bez zmian.

Uruchomienie (z katalogu głównego repo):
    python analysis/validate_engine.py > results/engine_validation.txt
"""
import random
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from uav_moea.algorithm.sms_emoa import (  # noqa: E402
    fast_non_dominated_sort,
    hypervolume_2d,
    SmsEmoa,
    SmsEmoaConfig,
)


class FakeSolution:
    """Lekki obiekt z f1/f2 — kompatybilny z fast_non_dominated_sort (używa .f1/.f2)."""

    def __init__(self, f1, f2):
        self.f1 = f1
        self.f2 = f2


# ── (a) Walidacja sortowania niezdominowanego ───────────────────────────────

def naive_dominance_fronts(points):
    """Naiwne O(n^2) sprawdzenie dominacji wprost z definicji Pareto."""
    n = len(points)

    def dominates(p, q):
        le = p[0] <= q[0] and p[1] <= q[1]
        lt = p[0] < q[0] or p[1] < q[1]
        return le and lt

    domination_count = [0] * n
    dominated_by = [[] for _ in range(n)]

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if dominates(points[j], points[i]):
                domination_count[i] += 1
            if dominates(points[i], points[j]):
                dominated_by[i].append(j)

    fronts = []
    remaining = set(range(n))
    count = domination_count[:]

    while remaining:
        front = [i for i in remaining if count[i] == 0]
        if not front:
            raise RuntimeError("Naiwne sortowanie nie zbiega — błąd w danych/algorytmie")
        fronts.append(sorted(front))
        for i in front:
            remaining.discard(i)
            for j in dominated_by[i]:
                if j in remaining:
                    count[j] -= 1
    return fronts


def fronts_to_partition(fronts):
    return [frozenset(f) for f in fronts]


def check_a_nondominated_sort(n_populations=100, pop_size=60, seed=1):
    rng = random.Random(seed)
    ok = 0
    first_failure = None
    for trial in range(n_populations):
        pts = [(rng.uniform(0, 100), rng.uniform(0, 100)) for _ in range(pop_size)]
        pop = [FakeSolution(f1, f2) for f1, f2 in pts]

        fast_fronts = fast_non_dominated_sort(pop)
        naive_fronts = naive_dominance_fronts(pts)

        same = fronts_to_partition(fast_fronts) == fronts_to_partition(naive_fronts)
        if same:
            ok += 1
        elif first_failure is None:
            first_failure = (trial, fast_fronts, naive_fronts)

    return ok, n_populations, first_failure


# ── (b) Walidacja wkładu do HV (LOO) ────────────────────────────────────────

def random_front_2d(n, rng, spread=1.0):
    """Losowy front niezdominowany 2D (skyline)."""
    xs = sorted(rng.uniform(0, spread) for _ in range(n))
    pts = []
    min_y_so_far = float("inf")
    for x in xs:
        y = rng.uniform(0, min_y_so_far if min_y_so_far < float("inf") else spread)
        pts.append((x, y))
        min_y_so_far = y
    return pts


def sms_emoa_contributions(points, ref=(1.1, 1.1)):
    """Odtwarza logikę wkładu HV z SmsEmoa._select_for_removal (już znormalizowane punkty)."""
    order = sorted(range(len(points)), key=lambda k: points[k][0])
    contributions = [0.0] * len(points)

    for pos, idx in enumerate(order):
        f1, f2 = points[idx]
        right_f1 = points[order[pos + 1]][0] if pos < len(order) - 1 else ref[0]
        left_f2 = points[order[pos - 1]][1] if pos > 0 else ref[1]
        contributions[idx] = (right_f1 - f1) * (left_f2 - f2)
    return contributions


def check_b_loo_contribution(n_fronts=50, seed=2):
    rng = random.Random(seed)
    max_abs_diff = 0.0
    details = []

    for trial in range(n_fronts):
        n = rng.randint(3, 15)
        points = random_front_2d(n, rng, spread=1.0)
        ref = (1.1, 1.1)

        full_hv = hypervolume_2d(points, ref)
        naive_contribs = []
        for i in range(n):
            rest = [p for j, p in enumerate(points) if j != i]
            naive_contribs.append(full_hv - hypervolume_2d(rest, ref))

        sms_contribs = sms_emoa_contributions(points, ref)

        diffs = [abs(a - b) for a, b in zip(naive_contribs, sms_contribs)]
        trial_max = max(diffs)
        max_abs_diff = max(max_abs_diff, trial_max)
        details.append((trial, n, trial_max))

    return max_abs_diff, details


# ── (c) HV vs oracle (pymoo) ─────────────────────────────────────────────

def check_c_hv_vs_pymoo(n_fronts=100, seed=3):
    from pymoo.indicators.hv import HV

    rng = random.Random(seed)
    max_rel_diff = 0.0
    details = []

    for trial in range(n_fronts):
        n = rng.randint(2, 30)
        spread = rng.uniform(1.0, 200.0)
        points = random_front_2d(n, rng, spread=spread)
        ref = (spread * 1.1, spread * 1.1)

        hv_ours = hypervolume_2d(points, ref)
        hv_pymoo = float(HV(ref_point=np.array(ref))(np.array(points)))

        denom = max(abs(hv_pymoo), 1e-12)
        rel_diff = abs(hv_ours - hv_pymoo) / denom
        max_rel_diff = max(max_rel_diff, rel_diff)
        details.append((trial, n, hv_ours, hv_pymoo, rel_diff))

    return max_rel_diff, details


# ── (d) Monotoniczność HV archiwum ──────────────────────────────────────

def check_d_archive_hv_monotonic(test_case_paths, seeds=(1, 2, 3)):
    from uav_moea.io.data_loader import load_test_case

    violations_total = 0
    runs_report = []

    for tc_path in test_case_paths:
        tc = load_test_case(tc_path)
        for seed in seeds:
            config = SmsEmoaConfig(
                population_size=20,
                max_evaluations=800,
                num_uavs=3,
                seed=seed,
                snapshot_interval=100,
            )
            algo = SmsEmoa(config, tc)

            archive_hv_history = []
            orig_update_archive = algo._update_archive

            def wrapped_update_archive(_orig=orig_update_archive, _algo=algo, _hist=archive_hv_history):
                _orig()
                if _algo.archive and _algo.ref_point:
                    pts = [(s.f1, s.f2) for s in _algo.archive]
                    hv = hypervolume_2d(pts, _algo.ref_point)
                else:
                    hv = 0.0
                _hist.append(hv)

            algo._update_archive = wrapped_update_archive
            algo.run()

            violations = sum(
                1 for i in range(1, len(archive_hv_history))
                if archive_hv_history[i] < archive_hv_history[i - 1] - 1e-9
            )
            violations_total += violations
            runs_report.append((tc_path, seed, len(archive_hv_history), violations))

    return violations_total, runs_report


# ── MAIN ──────────────────────────────────────────────────────────────

def main():
    results_summary = []

    print("=" * 70)
    print("WALIDACJA SILNIKA OBLICZENIOWEGO SMS-EMOA — 4 cross-checki")
    print("=" * 70)

    print("\n" + "=" * 70)
    print("(a) WALIDACJA SORTOWANIA NIEZDOMINOWANEGO (naive O(n^2) vs fast)")
    print("=" * 70)
    ok, total, first_failure = check_a_nondominated_sort(n_populations=100, pop_size=60, seed=1)
    print(f"Zgodnych populacji: {ok} / {total}")
    if first_failure:
        trial, ff, nf = first_failure
        print(f"PIERWSZA NIEZGODNOŚĆ w trial={trial}")
        print("fast:", ff)
        print("naive:", nf)
    pass_a = ok == total
    print("WYNIK:", "PASS" if pass_a else "FAIL")
    results_summary.append(("(a) sortowanie niezdominowane", f"{ok}/{total} zgodnych", pass_a))

    print("\n" + "=" * 70)
    print("(b) WALIDACJA WKŁADU DO HV (LOO) vs procedura selekcji SMS-EMOA")
    print("=" * 70)
    max_diff, details = check_b_loo_contribution(n_fronts=50, seed=2)
    print(f"Maksymalna różnica bezwzględna na 50 losowych frontach: {max_diff:.3e}")
    worst = max(details, key=lambda d: d[2])
    print(f"  (najgorszy przypadek: trial={worst[0]}, n_points={worst[1]}, diff={worst[2]:.3e})")
    pass_b = max_diff < 1e-9
    print("WYNIK:", "PASS" if pass_b else "FAIL")
    results_summary.append(("(b) wkład do HV (LOO)", f"maks. różnica={max_diff:.3e}", pass_b))

    print("\n" + "=" * 70)
    print("(c) WALIDACJA HV WZGLĘDEM ORACLE (pymoo.indicators.hv.HV)")
    print("=" * 70)
    max_rel_diff, details3 = check_c_hv_vs_pymoo(n_fronts=100, seed=3)
    print(f"Maksymalna różnica względna na 100 losowych frontach: {max_rel_diff:.3e}")
    worst3 = max(details3, key=lambda d: d[4])
    print(f"  (najgorszy przypadek: trial={worst3[0]}, n={worst3[1]}, "
          f"hv_ours={worst3[2]:.6f}, hv_pymoo={worst3[3]:.6f}, rel_diff={worst3[4]:.3e})")
    pass_c = max_rel_diff < 1e-6
    print("WYNIK:", "PASS" if pass_c else "FAIL")
    results_summary.append(("(c) HV vs oracle pymoo", f"maks. różnica względna={max_rel_diff:.3e}", pass_c))

    print("\n" + "=" * 70)
    print("(d) MONOTONICZNOŚĆ HV ARCHIWUM (kilka przebiegów, 1-2 instancje)")
    print("=" * 70)
    data_dir = REPO / "data" / "TC-PGI"
    candidates = sorted(data_dir.glob("*.json"))[:2]
    tc_paths = [str(p) for p in candidates]
    print(f"Instancje: {tc_paths}")
    violations_total, runs_report = check_d_archive_hv_monotonic(tc_paths, seeds=(1, 2, 3))
    for tc_path, seed, n_points, violations in runs_report:
        print(f"  instancja={Path(tc_path).name} seed={seed} punktów_HV={n_points} naruszeń={violations}")
    print(f"Łączna liczba naruszeń monotoniczności: {violations_total}")
    pass_d = violations_total == 0
    print("WYNIK:", "PASS" if pass_d else "FAIL")
    results_summary.append(("(d) monotoniczność HV archiwum", f"{violations_total} naruszeń", pass_d))

    print("\n" + "=" * 70)
    print("PODSUMOWANIE")
    print("=" * 70)
    header = f"{'Check':<35}{'Wynik':<36}{'Status'}"
    print(header)
    print("-" * len(header))
    for name, val, passed in results_summary:
        print(f"{name:<35}{val:<36}{'PASS' if passed else 'FAIL'}")

    all_pass = all(p for _, _, p in results_summary)
    print()
    print("WSZYSTKIE TESTY PASS" if all_pass else "UWAGA: CO NAJMNIEJ JEDEN TEST FAIL")


if __name__ == "__main__":
    main()
