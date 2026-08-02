#!/usr/bin/env python3
"""
Ablacja operatorow mutacji — greedy backward elimination na ZAMROZONYM
najlepszym dostrojonym configu (GEN-SMS-EMOA), BEZ przestrajania.

Styl Fawcett & Hoos (2016) "Analysing differences ... through ablation":
sciezka ablacji, na kazdym kroku usuwamy 1 operator (ten, ktorego usuniecie
najmniej szkodzi / najbardziej pomaga), i idziemy 6 -> 5 -> 4 -> ... -> 1.

Usuniecie operatora = jego waga -> 0 + RENORMALIZACJA reszty (bez re-tuningu).
Dzieki temu sciezka izoluje efekt USUNIECIA operatora, wolny od szumu irace
(w odroznieniu od starej, blednej wersji, ktora przestrajala per licznosc).

Baza: GEN-SMS-EMOA, irace #614 (najlepszy dostrojony wariant):
  pop=90, pc=0.2925, pm=0.9336,
  wagi: invert 0.9156, or_opt 0.7073, transfer 0.2751,
        swap 0.1578, cross 0.6379, exchange 0.0253.

Protokol jak w reszcie pracy: 30000 ewaluacji, 8 instancji x 30 ziaren,
HV liczone na GLOBALNYCH granicach (results/global_norms.json), ref (1.1,1.1).

UWAGA interpretacyjna: mierzymy WAZNOSC operatora WEWNATRZ dostrojonego
configu (jak w artykule), a nie "najlepszy mozliwy algorytm z k operatorami"
(bo wagi strojono lacznie; nie re-optymalizujemy podzbioru).

Tryby:
  --smoke   : tcW x 3 ziarna, tylko pierwszy poziom (6->5) — walidacja + pomiar czasu
  (default) : pelny greedy 6->1, 8 instancji x 30 ziaren
  --workers N : liczba procesow (default: liczba rdzeni)

Wyniki:
  results/ablation/<tag>_<inst>_s<seed>/results.json   (archiwa)
  results/ablation/ablation_path.json                  (sciezka + wybory)
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "analysis"))

from uav_moea.io.data_loader import load_test_case          # noqa: E402
from uav_moea.algorithm.gen_sms_emoa import GenSmsEmoa, GenSmsEmoaConfig  # noqa: E402
from hv_stats import load_global_norms, hv_normed_global   # noqa: E402

OPERATORS = ["invert", "or_opt", "transfer", "swap", "cross", "exchange"]
BASE_WEIGHTS = {
    "invert": 0.9156, "or_opt": 0.7073, "transfer": 0.2751,
    "swap": 0.1578, "cross": 0.6379, "exchange": 0.0253,
}
POP, PC, PM = 90, 0.2925, 0.9336
EVAL = 30_000

INSTANCES = {
    "tcB":  "data/TC-PGI/tcB.json",  "tcT":  "data/TC-PGI/tcT.json",
    "tcV":  "data/TC-PGI/tcV.json",  "tcW":  "data/TC-PGI/tcW.json",
    "tcGB": "data/TC-SP/tcGB.json",  "tcGL": "data/TC-SP/tcGL.json",
    "tcLO": "data/TC-SP/tcLO.json",  "tcSW": "data/TC-SP/tcSW.json",
}
# te same 30 ziaren co w pozostalych eksperymentach (random.Random(42))
import random as _r
_rng = _r.Random(42)
SEEDS = [_rng.randint(0, 1_000_000) for _ in range(30)]

OUTDIR = REPO / "results" / "ablation"


def weights_for(active_ops):
    """Wektor 6 wag: baza dla aktywnych operatorow, 0 dla usunietych, renormalizacja."""
    raw = [BASE_WEIGHTS[op] if op in active_ops else 0.0 for op in OPERATORS]
    total = sum(raw)
    return [w / total for w in raw] if total > 0 else [1.0 / len(raw)] * len(raw)


def tag_for(active_ops):
    removed = [op for op in OPERATORS if op not in active_ops]
    return "full" if not removed else "L%d_minus_%s" % (len(active_ops), "_".join(removed))


def run_one(active_ops, inst_name, seed):
    """Uruchamia GEN-SMS z wagami dla active_ops; zapisuje archiwum; zwraca (inst, seed, hv_global)."""
    weights = weights_for(active_ops)
    tc = load_test_case(str(REPO / INSTANCES[inst_name]))
    cfg = GenSmsEmoaConfig(
        population_size=POP, max_evaluations=EVAL, crossover_prob=PC,
        mutation_rate=PM, num_uavs=3, seed=seed, snapshot_interval=99999,
        mutation_weights=weights,
    )
    archive = GenSmsEmoa(cfg, tc).run()
    pts = [(s.f1, s.f2) for s in archive]

    tag = tag_for(active_ops)
    run_dir = OUTDIR / f"{tag}_{inst_name}_s{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "results.json").write_text(json.dumps({
        "algorithm": "gen_sms_emoa", "tag": tag, "active_ops": sorted(active_ops),
        "config": {"population_size": POP, "pc": PC, "pm": PM,
                   "max_evaluations": EVAL, "num_uavs": 3, "seed": seed,
                   "mutation_weights": weights},
        "archive": [{"f1": f1, "f2": f2} for f1, f2 in pts],
    }))

    norms = load_global_norms()[inst_name]
    return inst_name, seed, hv_normed_global(pts, norms)


def _worker(args):
    active_ops, inst_name, seed = args
    try:
        return run_one(set(active_ops), inst_name, seed)
    except Exception as e:  # noqa: BLE001
        return inst_name, seed, float("nan"), repr(e)


def evaluate(active_ops, instances, seeds, workers):
    """Rownolegle uruchom wszystkie (instancja, ziarno); zwroc dict inst -> lista HV."""
    tasks = [(sorted(active_ops), inst, s) for inst in instances for s in seeds]
    hv = {inst: [] for inst in instances}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_worker, t) for t in tasks]
        for f in as_completed(futs):
            res = f.result()
            inst, seed, val = res[0], res[1], res[2]
            hv[inst].append(val)
    return hv


def aggregate(hv):
    """Zwraca (mediana_per_instancja: dict, srednia_ogolna: float)."""
    med = {inst: float(np.nanmedian(vals)) for inst, vals in hv.items()}
    overall = float(np.nanmean(list(med.values())))
    return med, overall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="tcW x 3 ziarna, tylko 6->5")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args()

    instances = ["tcW"] if args.smoke else list(INSTANCES)
    seeds = SEEDS[:3] if args.smoke else SEEDS
    OUTDIR.mkdir(parents=True, exist_ok=True)

    print(f"Ablacja operatorow (baza GEN-SMS-EMOA #614), workers={args.workers}")
    print(f"  instancje={instances}  ziarna={len(seeds)}  eval={EVAL}")

    active = set(OPERATORS)
    path = []

    # Poziom 6 (pelny) — punkt odniesienia
    t0 = time.time()
    hv6 = evaluate(active, instances, seeds, args.workers)
    med6, ov6 = aggregate(hv6)
    print(f"\n[L6 full] HV_ogolne={ov6:.4f}  ({time.time()-t0:.0f}s)")
    path.append({"level": 6, "removed": None, "active": sorted(active),
                 "overall_hv": ov6, "median_per_inst": med6})

    # Greedy backward: 6->5->...->1 (smoke: tylko jeden krok)
    max_removals = 1 if args.smoke else len(OPERATORS) - 1
    for _ in range(max_removals):
        cand_results = []
        for op in sorted(active):
            cand = active - {op}
            hv_c = evaluate(cand, instances, seeds, args.workers)
            med_c, ov_c = aggregate(hv_c)
            cand_results.append((op, ov_c, med_c))
            print(f"  [L{len(cand)}] usun {op:9s} -> HV_ogolne={ov_c:.4f}")
        # wybierz usuniecie o NAJWYZSZYM HV (najmniejsza strata)
        best_op, best_ov, best_med = max(cand_results, key=lambda x: x[1])
        active = active - {best_op}
        print(f"  => usuwam '{best_op}' (HV={best_ov:.4f}); pozostaje {sorted(active)}")
        path.append({"level": len(active), "removed": best_op, "active": sorted(active),
                     "overall_hv": best_ov, "median_per_inst": best_med,
                     "all_candidates": {op: ov for op, ov, _ in cand_results}})

    (OUTDIR / "ablation_path.json").write_text(json.dumps(path, indent=2))
    print("\n=== SCIEZKA ABLACJI (HV_ogolne wg liczby operatorow) ===")
    for step in path:
        r = step["removed"] or "-"
        print(f"  L{step['level']}  usunieto={r:9s}  HV={step['overall_hv']:.4f}")
    print(f"\nZapisano: {(OUTDIR / 'ablation_path.json').relative_to(REPO)}")


if __name__ == "__main__":
    main()
