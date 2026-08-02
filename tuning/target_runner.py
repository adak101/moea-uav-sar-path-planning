#!/usr/bin/env python3
"""
Target runner dla irace — pełny tuning SMS-EMOA.
irace czyta stdout — wszystkie logi przekierowane na stderr.
"""
import sys
import os
import argparse
import logging

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

_real_stdout = sys.stdout
sys.stdout = sys.stderr
logging.disable(logging.CRITICAL)

from uav_moea.io.data_loader import load_test_case
from uav_moea.algorithm.sms_emoa import SmsEmoa, SmsEmoaConfig

sys.stdout = _real_stdout

# Stałe punkty referencyjne (nadir * 1.1) wyznaczone z 5 seedów per instancja.
# Pozwala porównywać HV między różnymi uruchomieniami i instancjami na wspólnej skali.
NADIRS = {
    "tcB":  (6275.13, 0.6127),
    "tcW":  (4879.85, 0.4959),
    "tcT":  (5961.93, 0.5842),
    "tcV":  (6016.48, 0.6151),
    "tcGB": (5655.68, 0.6669),
    "tcGL": (6624.80, 0.5283),
    "tcLO": (5335.96, 0.5749),
    "tcSW": (6098.32, 0.5787),
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("config_id")
    p.add_argument("instance_id")
    p.add_argument("seed", type=int)
    p.add_argument("instance_path")
    p.add_argument("bound", nargs="?")
    p.add_argument("--w_invert",   type=float, required=True)
    p.add_argument("--w_or_opt",   type=float, required=True)
    p.add_argument("--w_transfer", type=float, required=True)
    p.add_argument("--w_swap",     type=float, required=True)
    p.add_argument("--w_cross",    type=float, required=True)
    p.add_argument("--w_exchange", type=float, required=True)
    p.add_argument("--pop", type=int,   required=True)
    p.add_argument("--pc",  type=float, required=True)
    p.add_argument("--pm",  type=float, required=True)
    return p.parse_args()


def normalize(w_list):
    total = sum(w_list)
    return [w / total for w in w_list] if total > 0 else [1.0 / len(w_list)] * len(w_list)


def hv2d(pts, ref):
    """Dokładny HV 2D względem stałego punktu referencyjnego ref=(f1_ref, f2_ref)."""
    dominated = [(f1, f2) for f1, f2 in pts if f1 < ref[0] and f2 < ref[1]]
    if not dominated:
        return 0.0
    s = sorted(dominated, key=lambda p: p[0])
    hv, prev_f2 = 0.0, ref[1]
    for f1, f2 in s:
        if prev_f2 - f2 > 0:
            hv += (ref[0] - f1) * (prev_f2 - f2)
        prev_f2 = min(prev_f2, f2)
    return hv


def compute_hv(archive, nadir):
    """HV względem stałego punktu referencyjnego (nadir instancji)."""
    if not archive:
        return 0.0
    pts = [(s.f1, s.f2) for s in archive]
    return hv2d(pts, nadir)


def main():
    sys.stdout = sys.stderr
    logging.disable(logging.CRITICAL)

    args = parse_args()
    weights = normalize([args.w_invert, args.w_or_opt, args.w_transfer,
                         args.w_swap, args.w_cross, args.w_exchange])

    with open(args.instance_path.strip()) as f:
        json_path = os.path.join(PROJECT_ROOT, f.read().strip())

    instance_name = os.path.splitext(os.path.basename(json_path))[0]
    nadir = NADIRS.get(instance_name)
    if nadir is None:
        print(f"ERROR: brak nadir dla instancji '{instance_name}'", file=sys.stderr)
        sys.exit(1)

    tc = load_test_case(json_path)

    config = SmsEmoaConfig(
        population_size=args.pop,
        max_evaluations=8000,
        crossover_prob=args.pc,
        mutation_rate=args.pm,
        num_uavs=3,
        seed=args.seed,
        snapshot_interval=99999,
        mutation_weights=weights,
    )

    algo = SmsEmoa(config, tc)
    archive = algo.run()
    hv = compute_hv(archive, nadir)

    sys.stdout = _real_stdout
    print(-hv)


if __name__ == "__main__":
    main()
