"""
Główny punkt wejścia: SMS-EMOA / NSGA-II / SS-NSGA-II / GEN-SMS-EMOA
oraz warianty porównawcze tuningu operatorów:
  sms_emoa_default — 6 operatorów, wagi domyślne
  sms_emoa_tuned   — 6 operatorów, wagi z irace [López-Ibáñez et al. 2016]

Użycie:
    python main.py --algo sms_emoa
    python main.py --algo nsga2 --test-case data/TC-SP/tcGB.json --eval 20000
    python main.py --algo sms_emoa_tuned --eval 30000 --seed 42
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Pozwól uruchomić bez instalacji pakietu (`python main.py`): dołóż src/ do ścieżki.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import matplotlib.pyplot as plt

from uav_moea.io.data_loader import load_test_case
from uav_moea.io.logger import setup_logging
from uav_moea.algorithm.sms_emoa import SmsEmoa, SmsEmoaConfig
from uav_moea.algorithm.nsga2 import Nsga2, Nsga2Config
from uav_moea.algorithm.ss_nsga2 import SsNsga2, SsNsga2Config
from uav_moea.algorithm.gen_sms_emoa import GenSmsEmoa, GenSmsEmoaConfig

# ── Wagi operatorów mutacji ───────────────────────────────────
# Kolejność: invert, or_opt, transfer, swap, cross_route, exchange

# Wagi domyślne — uniform (brak wiedzy a priori o operatorach)
DEFAULT_WEIGHTS = [1/6, 1/6, 1/6, 1/6, 1/6, 1/6]

# Wagi z tuningu irace (konfiguracja #534, budżet 5000 uruchomień)
# Kolejność: invert, or_opt, transfer, swap, cross_route, exchange
# pop=108, pc=0.2786, pm=0.9388
IRACE_WEIGHTS = [0.8211, 0.2422, 0.3585, 0.0418, 0.7481, 0.0506]

# Wagi z tuningu irace dla NSGA-II (konfiguracja #550, budżet 5000 uruchomień)
# pop=92, pc=0.215, pm=0.9856
NSGA2_IRACE_WEIGHTS = [0.6837, 0.5387, 0.4616, 0.1599, 0.3946, 0.0660]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="UAV Path Optimization — wielokryterialny plan ablacyjny 2×2"
    )

    parser.add_argument(
        "--algo",
        type=str,
        default="sms_emoa",
        choices=[
            "sms_emoa",
            "nsga2",
            "ss_nsga2",
            "gen_sms_emoa",
            "sms_emoa_default",
            "sms_emoa_tuned",
            "sms_emoa_randinit",
            "nsga2_tuned",
            "nsga2_tuned_proper",
            "ss_nsga2_tuned",
            "gen_sms_emoa_tuned",
        ],
        help="Algorytm (default: sms_emoa)",
    )
    parser.add_argument(
        "--test-case",
        type=str,
        default="data/TC-PGI/tcB.json",
        help="Ścieżka do pliku JSON (default: data/TC-PGI/tcB.json)",
    )
    parser.add_argument("--uavs",      type=int,   default=3,      help="Liczba dronów (default: 3)")
    parser.add_argument("--pop",       type=int,   default=100,    help="Rozmiar populacji (default: 100)")
    parser.add_argument("--eval",      type=int,   default=10_000, help="Max ewaluacji (default: 10000)")
    parser.add_argument("--mutation",  type=float, default=1.0,    help="Mutation rate (default: 1.0)")
    parser.add_argument("--crossover", type=float, default=0.9,    help="Crossover prob (default: 0.9)")
    parser.add_argument("--seed",      type=int,   default=115986, help="Random seed (default: 115986)")
    parser.add_argument("--snapshot",  type=int,   default=500,    help="Snapshot interval (default: 500)")
    parser.add_argument("--no-plots",  action="store_true",        help="Nie pokazuj wykresów")
    parser.add_argument("--outdir",    type=str, default="results/runs", help="Katalog bazowy wyników")

    return parser.parse_args()


def plot_results(algo, output_dir: Path, show: bool = True) -> None:
    """Generuj wykresy: HV, best f1/f2, front Pareto."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    ax.plot(algo.history["generations"], algo.history["hypervolume"], color="steelblue")
    ax.set_xlabel("Generacja")
    ax.set_ylabel("Hypervolume")
    ax.set_title("Zbieżność Hypervolume")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(algo.history["generations"], algo.history["best_f1"], label="best f1", color="#e74c3c")
    ax.set_xlabel("Generacja")
    ax.set_ylabel("f1")
    ax.set_title("Najlepsze f1 / f2")
    ax.grid(True, alpha=0.3)

    ax2 = ax.twinx()
    ax2.plot(algo.history["generations"], algo.history["best_f2"], label="best f2", color="#2ecc71")
    ax2.set_ylabel("f2")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2)

    ax = axes[2]
    arch_f1 = [sol.f1 for sol in algo.archive]
    arch_f2 = [sol.f2 for sol in algo.archive]
    ax.scatter(arch_f1, arch_f2, c="steelblue", edgecolors="black", s=60, zorder=3)
    ax.set_xlabel("f1 — Czas zakończenia misji")
    ax.set_ylabel("f2 — Współczynnik nieodkrytego obszaru")
    ax.set_title(f"Front Pareto ({len(algo.archive)} rozwiązań)")
    ax.grid(True, alpha=0.3)

    if isinstance(algo, SmsEmoa):
        algo_name = "SMS-EMOA"
    elif isinstance(algo, SsNsga2):
        algo_name = "SS-NSGA-II"
    elif isinstance(algo, GenSmsEmoa):
        algo_name = "GEN-SMS-EMOA"
    else:
        algo_name = "NSGA-II"

    fig.suptitle(
        f"{algo_name}: {algo.test_case.name}, {algo.config.num_uavs} UAV, "
        f"{algo.config.max_evaluations} ewaluacji, seed={algo.config.seed}",
        fontsize=14,
    )
    fig.tight_layout()

    save_path = output_dir / "results.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Wykres: {save_path}")

    if show:
        plt.show()
    plt.close()


def main():
    args = parse_args()

    tc_name = Path(args.test_case).stem
    seed_str = f"_s{args.seed}" if args.seed is not None else ""
    run_name = f"{args.algo}_{tc_name}_u{args.uavs}_p{args.pop}_e{args.eval}{seed_str}"

    output_dir = Path(args.outdir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(log_file=output_dir / "log.txt")
    tc = load_test_case(args.test_case)

    common = dict(
        population_size=args.pop,
        max_evaluations=args.eval,
        num_uavs=args.uavs,
        crossover_prob=args.crossover,
        mutation_rate=args.mutation,
        seed=args.seed,
        snapshot_interval=args.snapshot,
    )

    if args.algo == "sms_emoa":
        algo = SmsEmoa(SmsEmoaConfig(**common), tc)

    elif args.algo == "sms_emoa_default":
        # SMS-EMOA z 6 operatorami i wagami domyślnymi (baseline tuningu)
        algo = SmsEmoa(SmsEmoaConfig(**common, mutation_weights=DEFAULT_WEIGHTS), tc)

    elif args.algo == "sms_emoa_tuned":
        # SMS-EMOA z 6 operatorami i wagami z irace (konfiguracja #52)
        algo = SmsEmoa(SmsEmoaConfig(**common, mutation_weights=IRACE_WEIGHTS), tc)

    elif args.algo == "sms_emoa_randinit":
        # SMS-EMOA z czystą losową inicjalizacją (punkt odniesienia dla hybrydy)
        algo = SmsEmoa(SmsEmoaConfig(**common, init_strategy="random"), tc)

    elif args.algo == "ss_nsga2":
        algo = SsNsga2(SsNsga2Config(**common), tc)

    elif args.algo == "gen_sms_emoa":
        algo = GenSmsEmoa(GenSmsEmoaConfig(**common), tc)

    elif args.algo == "nsga2_tuned":
        algo = Nsga2(Nsga2Config(**common, mutation_weights=IRACE_WEIGHTS), tc)

    elif args.algo == "nsga2_tuned_proper":
        # NSGA-II z własnym tuningiem irace (konfiguracja #550, budżet 5000 uruchomień)
        algo = Nsga2(Nsga2Config(**common, mutation_weights=NSGA2_IRACE_WEIGHTS), tc)

    elif args.algo == "ss_nsga2_tuned":
        algo = SsNsga2(SsNsga2Config(**common, mutation_weights=IRACE_WEIGHTS), tc)

    elif args.algo == "gen_sms_emoa_tuned":
        algo = GenSmsEmoa(GenSmsEmoaConfig(**common, mutation_weights=IRACE_WEIGHTS), tc)

    else:  # nsga2
        algo = Nsga2(Nsga2Config(**common), tc)

    algo.run()
    algo.save_results(output_dir / "results.json")
    plot_results(algo, output_dir, show=not args.no_plots)

    print(f"\nWyniki w: {output_dir}")


if __name__ == "__main__":
    main()