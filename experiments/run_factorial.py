"""
Batch runner — Faza 1: porównanie bazowe.

4 algorytmy na jednakowych parametrach domyślnych:
  pop=100, pc=0.6, pm=1.0, wagi operatorów=uniform
  pc=0.6 za: Prins (2004), Potvin & Bengio (1996) — OX crossover dla permutacji
  30 seedów × 8 instancji = 960 rund

Cel: ustalenie punktu odniesienia przed tuningiem.
"""
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

INSTANCES = {
    "tcB":  "data/TC-PGI/tcB.json",
    "tcT":  "data/TC-PGI/tcT.json",
    "tcV":  "data/TC-PGI/tcV.json",
    "tcW":  "data/TC-PGI/tcW.json",
    "tcGB": "data/TC-SP/tcGB.json",
    "tcGL": "data/TC-SP/tcGL.json",
    "tcLO": "data/TC-SP/tcLO.json",
    "tcSW": "data/TC-SP/tcSW.json",
}

# 30 seedów wygenerowanych deterministycznie (random.Random(42))
SEEDS = [
    670488, 116740,  26226, 777573, 288390,
    256788, 234054, 146317, 772247, 107474,
    709571, 776647, 935519, 571859,  91162,
    619177, 442418,  33327,  31245,  98247,
    229259, 243963, 529904, 631263,  27825,
    588509, 208497, 750801, 681454, 735393,
]

EVAL_BUDGET = 30_000
WORKERS = 4
OUTDIR = "results/factorial"

# Faza 1 v2: pc=0.6 zgodnie z literaturą VRP/kombinatoryczną
# (Prins 2004, Potvin & Bengio 1996 — OX crossover dla permutacji)
VARIANTS = {
    "sms_emoa":     dict(pop=100, crossover=0.6, mutation=1.0),
    "nsga2":        dict(pop=100, crossover=0.6, mutation=1.0),
    "ss_nsga2":     dict(pop=100, crossover=0.6, mutation=1.0),
    "gen_sms_emoa": dict(pop=100, crossover=0.6, mutation=1.0),
}


def run_one(algo, tc_name, tc_path, seed, pop, crossover, mutation):
    t0 = time.time()
    r = subprocess.run(
        [
            "python", "-u", "main.py",
            "--algo",      algo,
            "--test-case", tc_path,
            "--seed",      str(seed),
            "--eval",      str(EVAL_BUDGET),
            "--pop",       str(pop),
            "--crossover", str(crossover),
            "--mutation",  str(mutation),
            "--uavs",      "3",
            "--outdir",    OUTDIR,
            "--no-plots",
        ],
        capture_output=True,
        text=True,
    )
    elapsed = time.time() - t0
    return algo, tc_name, seed, r.returncode, elapsed


def main():
    jobs = [
        (algo, tc_name, tc_path, seed, p["pop"], p["crossover"], p["mutation"])
        for algo, p in VARIANTS.items()
        for tc_name, tc_path in INSTANCES.items()
        for seed in SEEDS
    ]
    total = len(jobs)
    print(f"Łącznie zadań: {total}  ({len(VARIANTS)} warianty × "
          f"{len(INSTANCES)} instancji × {len(SEEDS)} seedów)")
    print(f"Workerów: {WORKERS}\n")

    done = 0
    failed = []
    batch_start = time.time()

    with ProcessPoolExecutor(max_workers=WORKERS) as executor:
        futures = {executor.submit(run_one, *job): job for job in jobs}
        for future in as_completed(futures):
            algo, tc_name, seed, code, elapsed = future.result()
            done += 1
            status = "OK" if code == 0 else f"BŁĄD (code={code})"
            print(f"[{done:>4}/{total}] {algo:<20} {tc_name:<6} s={seed:<7} "
                  f"{elapsed:>6.1f}s  {status}", flush=True)
            if code != 0:
                failed.append((algo, tc_name, seed))

    total_time = time.time() - batch_start
    print(f"\n{'='*60}")
    print(f"Skończone w {total_time/60:.1f} min")
    print(f"Sukcesów: {done - len(failed)}/{total}")
    if failed:
        print("Niepowodzenia:")
        for algo, tc, seed in failed:
            print(f"  - {algo} {tc} seed={seed}")


if __name__ == "__main__":
    main()
