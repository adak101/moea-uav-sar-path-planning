#!/usr/bin/env python3
"""Rysunek sciezki ablacji operatorow (HV vs liczba operatorow) — prosty styl pracy."""
import json
from pathlib import Path

import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10,
})

PL = {"invert": "invert", "or_opt": "or-opt", "transfer": "transfer",
      "swap": "swap", "cross": "cross-route", "exchange": "exchange"}


def main():
    path = json.loads((REPO / "results/ablation/ablation_path.json").read_text())
    n = [s["level"] for s in path]
    hv = [s["overall_hv"] for s in path]
    removed = [s["removed"] for s in path]
    full_hv = hv[0]
    peak_i = max(range(len(hv)), key=lambda i: hv[i])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axhline(full_hv, color="gray", linestyle=":", linewidth=1,
               label="pełny zestaw (6 operatorów)")
    ax.plot(n, hv, "-o", color="#1f6f4a", linewidth=2, markersize=7, zorder=3)
    # wyroznienie szczytu
    ax.plot(n[peak_i], hv[peak_i], "*", color="#c0392b", markersize=18, zorder=4,
            label="najlepszy zestaw")

    # podpis usunietego operatora przy kazdym punkcie (poza pelnym)
    for i in range(1, len(n)):
        ax.annotate(f"–{PL.get(removed[i], removed[i])}",
                    (n[i], hv[i]), textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=9, color="#333333")

    ax.set_xlabel("Liczba operatorów mutacji")
    ax.set_ylabel("Znormalizowane $HV$ (średnia po instancjach)")
    ax.set_title("Ablacja operatorów — ścieżka greedy\n(baza: GEN-SMS-EMOA dostrojony)")
    ax.set_xticks(n)
    ax.invert_xaxis()  # 6 po lewej, 1 po prawej (usuwanie operatorow w prawo)
    ax.grid(alpha=0.3, linestyle="--")
    ax.legend(loc="lower left")
    ax.margins(y=0.12)

    for ext in ("pdf", "png"):
        fig.savefig(REPO / f"results/thesis_ablation_path.{ext}", dpi=150,
                    bbox_inches="tight")
    # kopia do figures/
    import shutil
    shutil.copy(REPO / "results/thesis_ablation_path.pdf",
                REPO / "folder z magisterką/figures/thesis_ablation_path.pdf")
    print("Zapisano thesis_ablation_path.pdf/.png (+ figures/)")


if __name__ == "__main__":
    main()
