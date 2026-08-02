"""
Ilustracje do pracy magisterskiej:
  1. Diagram reprezentacji rozwiązania — nieregularna siatka regionów z trasami
  2. Porównanie frontów Pareto: SMS-EMOA vs NSGA-II (instancja tcT)

Konwergencja (plot_convergence_init) liczy znormalizowany HV na wspólnych
globalnych granicach (results/global_norms.json) — tak samo jak tabele
w Rozdziale 4 — a nie surowe HV podzielone przez maksimum, jak wcześniej
(patrz audyt_spojnosci.md, A3).

Uruchomienie: python analysis/plot_illustrations.py
Wyniki: results/thesis_illustration_*.pdf / .png
"""
import json
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "analysis"))
from hv_stats import load_global_norms, hv_normed_global  # noqa: E402

OUTDIR = Path("results")

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "figure.dpi": 100,
})


def save_fig(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(OUTDIR / f"{name}.{ext}", dpi=150, bbox_inches="tight")
    print(f"  Zapisano {name}.pdf / .png")


# ── Wykres 1 — Diagram reprezentacji rozwiązania (nieregularna siatka) ──────

def draw_arrow(ax, p1, p2, color, rad=0.15, lw=2.0):
    ax.annotate(
        "", xy=p2, xytext=p1,
        arrowprops=dict(
            arrowstyle="-|>",
            color=color,
            lw=lw,
            connectionstyle=f"arc3,rad={rad}",
            mutation_scale=13,
        ),
        zorder=5,
    )


def plot_solution_diagram():
    print("Wykres 1: Diagram reprezentacji rozwiązania (kółka + strzałki)...")

    depot = np.array([0.12, 0.50])

    # UAV 1 — trasa górna
    uav1_pts = np.array([
        [0.32, 0.80],
        [0.55, 0.88],
        [0.74, 0.78],
        [0.52, 0.62],
    ])
    # UAV 2 — trasa prawa
    uav2_pts = np.array([
        [0.78, 0.52],
        [0.90, 0.38],
        [0.78, 0.22],
        [0.60, 0.35],
    ])
    # UAV 3 — trasa dolna
    uav3_pts = np.array([
        [0.35, 0.35],
        [0.50, 0.18],
        [0.68, 0.12],
        [0.82, 0.10],
    ])

    UAV_COLORS = ["#CD4B4B", "#2271B2", "#2CA02C"]
    UAV_LABELS = ["UAV 1", "UAV 2", "UAV 3"]
    UAV_ROUTES = [uav1_pts, uav2_pts, uav3_pts]
    RADS = [0.18, -0.15, 0.12]

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.05, 1.02)
    ax.set_aspect("equal")
    ax.axis("off")

    # ── Strzałki tras ──
    for route, color, rad in zip(UAV_ROUTES, UAV_COLORS, RADS):
        draw_arrow(ax, depot, route[0], color, rad=rad * 0.6)
        for i in range(len(route) - 1):
            draw_arrow(ax, route[i], route[i + 1], color, rad=rad)
        draw_arrow(ax, route[-1], depot, color, rad=rad * 0.6)

    # ── Depot ──
    ax.plot(*depot, marker="*", markersize=22, color="#333333",
            zorder=6, markeredgecolor="white", markeredgewidth=0.8)
    ax.text(depot[0], depot[1] - 0.07, "Baza", ha="center", va="top",
            fontsize=11, fontweight="bold", color="#333333")

    # ── Kółka z numerami regionów ──
    region_num = 1
    for route, color in zip(UAV_ROUTES, UAV_COLORS):
        for pt in route:
            circle = plt.Circle(pt, 0.045, color=color, zorder=5, alpha=0.9,
                                linewidth=1.5, ec="white")
            ax.add_patch(circle)
            ax.text(pt[0], pt[1], str(region_num), ha="center", va="center",
                    fontsize=10, fontweight="bold", color="white", zorder=7)
            region_num += 1

    # ── Legenda (bez depot) ──
    legend_patches = [
        mpatches.Patch(color=c, label=l, alpha=0.9)
        for c, l in zip(UAV_COLORS, UAV_LABELS)
    ]
    ax.legend(handles=legend_patches, loc="upper left",
              fontsize=10, framealpha=0.9)

    ax.set_title(
        "Reprezentacja rozwiązania: 3 drony, 12 regionów\n"
        "Każda trasa startuje w bazie (powrót nie wlicza się do czasu misji)",
        fontsize=12, pad=10,
    )

    fig.tight_layout()
    save_fig(fig, "thesis_illustration_solution")
    plt.close(fig)


# ── Wykres 2 — Porównanie frontów Pareto: SMS-EMOA vs NSGA-II ───────────────

def load_pareto(path):
    d = json.loads(path.read_text())
    pts = sorted([(s["f1"], s["f2"]) for s in d["archive"]])
    return [p[0] for p in pts], [p[1] for p in pts]


def plot_pareto_comparison():
    print("Wykres 2: Porównanie frontów Pareto (SMS-EMOA domyślny vs irace, tcGB)...")

    BASE1 = Path("results/factorial")
    BASE2 = Path("results/tuning")
    INST  = "tcGB"
    SEED  = 670488

    VARIANTS = [
        (BASE1 / f"sms_emoa_{INST}_u3_p100_e30000_s{SEED}/results.json",
         "#2271B2", "o", "SMS-EMOA (domyślny)"),
        (BASE2 / f"sms_emoa_tuned_{INST}_u3_p108_e30000_s{SEED}/results.json",
         "#8B1A8B", "*", "SMS-EMOA (irace)"),
    ]

    fig, ax = plt.subplots(figsize=(8, 5.5))

    all_f1, all_f2 = [], []
    for path, color, marker, label in VARIANTS:
        f1s, f2s = load_pareto(path)
        sz = 55 if marker == "*" else 28
        ax.scatter(f1s, f2s, color=color, marker=marker, s=sz,
                   zorder=4, alpha=0.85, label=f"{label} (n={len(f1s)})")
        ax.plot(f1s, f2s, color=color, alpha=0.40, linewidth=1.4, zorder=3)
        all_f1.extend(f1s); all_f2.extend(f2s)

    # Jawne limity z marginesem 5% po każdej stronie
    f1_pad = (max(all_f1) - min(all_f1)) * 0.05
    f2_pad = (max(all_f2) - min(all_f2)) * 0.05
    ax.set_xlim(min(all_f1) - f1_pad, max(all_f1) + f1_pad)
    ax.set_ylim(min(all_f2) - f2_pad, max(all_f2) + f2_pad)

    ax.set_xlabel("$f_1$ — czas zakończenia misji [j.u.]", fontsize=11)
    ax.set_ylabel("$f_2$ — wsp. nieodkrytego obszaru", fontsize=11)
    ax.set_title(
        f"Fronty Pareto — SMS-EMOA domyślny vs dostrojony (irace)\n"
        f"instancja {INST}, ziarno {SEED}",
        fontsize=12,
    )
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(fontsize=10, loc="upper right", framealpha=0.9)

    fig.tight_layout()
    save_fig(fig, "thesis_illustration_pareto")
    plt.close(fig)


# ── Wykres 3 — 4 fronty Pareto: porównanie architektur algorytmów ───────────

ALGO_COLORS = {
    "sms_emoa":     "#2271B2",
    "nsga2":        "#CD4B4B",
    "gen_sms_emoa": "#2CA02C",
    "ss_nsga2":     "#F28500",
}
ALGO_LABELS = {
    "sms_emoa":     "SMS-EMOA (steady-state, HV)",
    "nsga2":        "NSGA-II (generacyjny, crowding)",
    "gen_sms_emoa": "GEN-SMS-EMOA (generacyjny, HV)",
    "ss_nsga2":     "SS-NSGA-II (steady-state, crowding)",
}
ALGO_MARKERS = {
    "sms_emoa":     "o",
    "nsga2":        "s",
    "gen_sms_emoa": "^",
    "ss_nsga2":     "D",
}


def plot_pareto_4algos():
    print("Wykres 3: 4 fronty Pareto — porównanie architektur (tcGB)...")

    BASE = Path("results/factorial")
    INST = "tcGB"
    SEED = 670488

    ALGOS = ["sms_emoa", "nsga2", "gen_sms_emoa", "ss_nsga2"]

    fig, ax = plt.subplots(figsize=(9, 5.8))

    all_f1, all_f2 = [], []
    for algo in ALGOS:
        path = BASE / f"{algo}_{INST}_u3_p100_e30000_s{SEED}/results.json"
        f1s, f2s = load_pareto(path)
        color  = ALGO_COLORS[algo]
        marker = ALGO_MARKERS[algo]
        label  = ALGO_LABELS[algo]
        ax.scatter(f1s, f2s, color=color, marker=marker, s=30,
                   zorder=4, alpha=0.85, label=f"{label} (n={len(f1s)})")
        ax.plot(f1s, f2s, color=color, alpha=0.40, linewidth=1.3, zorder=3)
        all_f1.extend(f1s); all_f2.extend(f2s)

    f1_pad = (max(all_f1) - min(all_f1)) * 0.05
    f2_pad = (max(all_f2) - min(all_f2)) * 0.05
    ax.set_xlim(min(all_f1) - f1_pad, max(all_f1) + f1_pad)
    ax.set_ylim(min(all_f2) - f2_pad, max(all_f2) + f2_pad)

    ax.set_xlabel("$f_1$ — czas zakończenia misji [j.u.]", fontsize=11)
    ax.set_ylabel("$f_2$ — wsp. nieodkrytego obszaru", fontsize=11)
    ax.set_title(
        f"Fronty Pareto — 4 algorytmy, instancja {INST}, ziarno {SEED}",
        fontsize=12,
    )
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(fontsize=9, loc="upper right", framealpha=0.9)

    fig.tight_layout()
    save_fig(fig, "thesis_illustration_4algos")
    plt.close(fig)


# ── Wykres 4 — Konwergencja: hybryda vs losowa inicjalizacja (tcGB) ─────────

def plot_convergence_init():
    print("Wykres 4: Konwergencja hybryda vs losowa inicjalizacja (tcGB)...")

    BASE_RAND = Path("results/initialization")
    BASE_HYB  = Path("results/factorial")
    INST = "tcGB"
    N_EVALS = 30_000
    GRID = np.arange(0, N_EVALS + 1, 500)

    norms_inst = load_global_norms()[INST]

    def load_hv_curves(base, pattern):
        curves = []
        for p in sorted(base.glob(f"{pattern}/results.json")):
            d = json.loads(p.read_text())
            snaps = d["history"]["snapshots"]
            evals_snap, hv_snap = [], []
            for s in snaps:
                pop_pts = [(pt["f1"], pt["f2"]) for pt in s["population"]]
                evals_snap.append(s["generation"])
                hv_snap.append(hv_normed_global(pop_pts, norms_inst))
            if len(evals_snap) < 2:
                continue
            curves.append(np.interp(GRID, np.array(evals_snap, float),
                                     np.array(hv_snap, float)))
        return np.array(curves)

    rand_mat = load_hv_curves(BASE_RAND, f"sms_emoa_randinit_{INST}_*")
    hyb_mat  = load_hv_curves(BASE_HYB,  f"sms_emoa_{INST}_*")
    evals = GRID

    def band(mat):
        return (np.median(mat, axis=0),
                np.percentile(mat, 25, axis=0),
                np.percentile(mat, 75, axis=0))

    hyb_med,  hyb_q1,  hyb_q3  = band(hyb_mat)
    rand_med, rand_q1, rand_q3 = band(rand_mat)

    fig, ax = plt.subplots(figsize=(8, 5))

    # Hybryda — zielony
    ax.fill_between(evals, hyb_q1, hyb_q3, alpha=0.18, color="#2CA02C")
    ax.plot(evals, hyb_med, color="#2CA02C", linewidth=2.0,
            label="Inicjalizacja hybrydowa")

    # Losowa — szary
    ax.fill_between(evals, rand_q1, rand_q3, alpha=0.18, color="#888888")
    ax.plot(evals, rand_med, color="#888888", linewidth=2.0,
            label="Inicjalizacja losowa")

    ax.set_xlabel("Liczba ewaluacji", fontsize=11)
    ax.set_ylabel("Znormalizowany HV", fontsize=11)
    ax.set_xlim(0, evals[-1])
    ax.set_ylim(0, None)
    ax.set_title(
        f"Konwergencja SMS-EMOA — inicjalizacja hybrydowa vs losowa\n"
        f"instancja {INST}, mediana ± IQR (30 ziaren)",
        fontsize=12,
    )
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(fontsize=10, loc="lower right", framealpha=0.9)

    fig.tight_layout()
    save_fig(fig, "thesis_illustration_convergence_init")
    plt.close(fig)


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=== Generowanie ilustracji ===\n")
    OUTDIR.mkdir(exist_ok=True)
    plot_solution_diagram()
    plot_pareto_comparison()
    plot_pareto_4algos()
    plot_convergence_init()
    print("\nGotowe.")


if __name__ == "__main__":
    main()
