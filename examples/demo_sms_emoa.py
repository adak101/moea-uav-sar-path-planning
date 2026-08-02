"""
Szybki test SMS-EMOA z wizualizacją.

Odpal: python tests/test_sms_emoa_run.py
"""
from pathlib import Path

import matplotlib.pyplot as plt

from uav_moea.io.data_loader import load_test_case
from uav_moea.io.logger import setup_logging
from uav_moea.algorithm.sms_emoa import SmsEmoa, SmsEmoaConfig


TC_PATH = "data/TC-PGI/tcB.json"


def main():
    setup_logging()

    tc = load_test_case(TC_PATH)

    config = SmsEmoaConfig(
        population_size=100,
        max_evaluations=10000,
        num_uavs=3,
        seed=42,
        snapshot_interval=500,
    )

    algo = SmsEmoa(config, tc)
    archive = algo.run()

    # Zapisz wyniki
    algo.save_results("results/runs/test_run.json")

    # ── Wykresy ──────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Hypervolume po generacjach
    ax = axes[0]
    ax.plot(algo.history["generations"], algo.history["hypervolume"], color="steelblue")
    ax.set_xlabel("Generacja")
    ax.set_ylabel("Hypervolume")
    ax.set_title("Konwergencja HV")
    ax.grid(True, alpha=0.3)

    # 2. Best f1 i f2 po generacjach
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

    # 3. Front Pareto (archiwum)
    ax = axes[2]
    arch_f1 = [sol.f1 for sol in archive]
    arch_f2 = [sol.f2 for sol in archive]
    ax.scatter(arch_f1, arch_f2, c="steelblue", edgecolors="black", s=60, zorder=3)
    ax.set_xlabel("f1 — Czas zakończenia misji")
    ax.set_ylabel("f2 — Współczynnik nieodkrytego obszaru")
    ax.set_title(f"Front Pareto ({len(archive)} rozwiązań)")
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"SMS-EMOA: {tc.name}, {config.num_uavs} UAV, "
        f"{config.max_evaluations} ewaluacji",
        fontsize=14,
    )
    fig.tight_layout()

    output_dir = Path("results/plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "sms_emoa_test_run.png", dpi=150, bbox_inches="tight")
    print(f"\nWykres: results/plots/sms_emoa_test_run.png")

    plt.show()


if __name__ == "__main__":
    main()