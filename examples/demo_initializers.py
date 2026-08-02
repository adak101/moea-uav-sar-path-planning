"""
Wizualizacja hybrydowej inicjalizacji — skąd pochodzi każde rozwiązanie.

Odpal: python tests/test_initializers.py
"""
from pathlib import Path

from uav_moea.io.data_loader import load_test_case
from uav_moea.model.objectives import evaluate
from uav_moea.algorithm.initializers import (
    generate_nearest_neighbor,
    generate_cluster_based,
    generate_victim_priority,
    generate_balanced_random,
    _perturb,
)

import matplotlib.pyplot as plt


TC_PATH = "data/TC-PGI/tcB.json"
NUM_UAVS = 3
POPULATION_SIZE = 50


def main():
    tc = load_test_case(TC_PATH)

    nn_fraction = 0.20
    cluster_fraction = 0.10
    victim_fraction = 0.10

    nn_count = max(1, int(POPULATION_SIZE * nn_fraction))
    cluster_count = max(1, int(POPULATION_SIZE * cluster_fraction))
    victim_count = max(1, int(POPULATION_SIZE * victim_fraction))
    balanced_count = POPULATION_SIZE - nn_count - cluster_count - victim_count

    strategies = [
        ("Nearest-Neighbor", generate_nearest_neighbor, nn_count, True),
        ("Cluster-Based", generate_cluster_based, cluster_count, True),
        ("Victim-Priority", generate_victim_priority, victim_count, True),
        ("Balanced-Random", generate_balanced_random, balanced_count, False),
    ]

    colors = {
        "Nearest-Neighbor": "#e74c3c",
        "Cluster-Based": "#3498db",
        "Victim-Priority": "#2ecc71",
        "Balanced-Random": "#95a5a6",
    }

    markers = {
        "Nearest-Neighbor": "o",
        "Cluster-Based": "s",
        "Victim-Priority": "^",
        "Balanced-Random": "x",
    }

    fig, ax = plt.subplots(figsize=(10, 7))

    print("\n" + "=" * 70)
    print(f"Hybrid Population: {POPULATION_SIZE} rozwiązań, {NUM_UAVS} UAV, {tc.name}")
    print("=" * 70)
    print(f"{'Strategia':<22} {'Ilość':>6} {'f1 (avg)':>10} {'f2 (avg)':>10}")
    print("-" * 70)

    for name, gen, count, perturb in strategies:
        f1_values = []
        f2_values = []

        for _ in range(count):
            sol = gen(tc, NUM_UAVS)
            if perturb:
                sol = _perturb(sol, tc)
            f1, f2 = evaluate(sol, tc)
            f1_values.append(f1)
            f2_values.append(f2)

        ax.scatter(
            f1_values,
            f2_values,
            c=colors[name],
            marker=markers[name],
            label=f"{name} ({count})",
            s=80,
            alpha=0.7,
            edgecolors="black",
            linewidths=0.5,
            zorder=3,
        )

        avg_f1 = sum(f1_values) / len(f1_values)
        avg_f2 = sum(f2_values) / len(f2_values)
        print(f"{name:<22} {count:>6} {avg_f1:>10.2f} {avg_f2:>10.6f}")

    print("=" * 70)

    ax.set_xlabel("f1 — Czas zakończenia misji", fontsize=12)
    ax.set_ylabel("f2 — Współczynnik nieodkrytego obszaru", fontsize=12)
    ax.set_title(
        f"Hybrydowa inicjalizacja populacji\n"
        f"{tc.name}, {NUM_UAVS} UAV, populacja = {POPULATION_SIZE}",
        fontsize=14,
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    output_dir = Path("results/plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / "initialization_comparison.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nWykres zapisany: {save_path}")

    plt.show()


if __name__ == "__main__":
    main()