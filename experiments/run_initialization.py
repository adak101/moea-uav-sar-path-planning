"""
Batch runner — Faza 0: uzasadnienie inicjalizacji hybrydowej.

sms_emoa_randinit na jednakowych parametrach co Faza 1:
  pop=100, pc=0.6, pm=1.0, wagi operatorów=uniform
  30 seedów × 8 instancji = 240 rund

Dane hybrydy (sms_emoa) pochodzą z Fazy 1 (results/factorial).
Wyniki randinit zapisywane do results/initialization.
"""
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

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
OUTDIR = "results/initialization"

POP = 100
CROSSOVER = 0.6
MUTATION = 1.0


def run_one(tc_name, tc_path, seed):
    t0 = time.time()
    r = subprocess.run(
        [
            "python", "-u", "main.py",
            "--algo",      "sms_emoa_randinit",
            "--test-case", tc_path,
            "--seed",      str(seed),
            "--eval",      str(EVAL_BUDGET),
            "--pop",       str(POP),
            "--crossover", str(CROSSOVER),
            "--mutation",  str(MUTATION),
            "--uavs",      "3",
            "--outdir",    OUTDIR,
            "--no-plots",
        ],
        capture_output=True,
        text=True,
    )
    elapsed = time.time() - t0
    return tc_name, seed, r.returncode, elapsed


def main():
    jobs = [
        (tc_name, tc_path, seed)
        for tc_name, tc_path in INSTANCES.items()
        for seed in SEEDS
    ]
    total = len(jobs)
    print(f"Faza 0 — sms_emoa_randinit")
    print(f"Łącznie zadań: {total}  (8 instancji × 30 seedów)")
    print(f"Workerów: {WORKERS}\n")

    done = 0
    failed = []
    batch_start = time.time()

    with ProcessPoolExecutor(max_workers=WORKERS) as executor:
        futures = {executor.submit(run_one, *job): job for job in jobs}
        for future in as_completed(futures):
            tc_name, seed, code, elapsed = future.result()
            done += 1
            status = "OK" if code == 0 else f"BŁĄD (code={code})"
            print(f"[{done:>4}/{total}] sms_emoa_randinit  {tc_name:<6} s={seed:<7} "
                  f"{elapsed:>6.1f}s  {status}", flush=True)
            if code != 0:
                failed.append((tc_name, seed))

    total_time = time.time() - batch_start
    print(f"\n{'='*60}")
    print(f"Skończone w {total_time/60:.1f} min")
    print(f"Sukcesów: {done - len(failed)}/{total}")
    if failed:
        print("Niepowodzenia:")
        for tc, seed in failed:
            print(f"  - {tc} seed={seed}")


if __name__ == "__main__":
    main()
