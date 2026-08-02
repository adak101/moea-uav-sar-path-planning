"""
Minimalne, wspólne narzędzia do hiperobjętości (HV) używane przez skrypty rysujące.

Granice normalizacji są policzone raz (globalnie, po wszystkich archiwach z pracy)
i zamrożone w ``results/global_norms.json``. Wszystkie figury i tabele w pracy
używają tych samych granic, dzięki czemu wartości HV są porównywalne między
eksperymentami. Punkt odniesienia HV: (1.1, 1.1).
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
REF = (1.1, 1.1)


def non_dominated(points):
    """Front Pareto (minimalizacja obu kryteriów) metodą przeglądu skyline."""
    pts = sorted(set(points))
    front, min_f2 = [], float("inf")
    for f1, f2 in pts:
        if f2 < min_f2:
            front.append((f1, f2))
            min_f2 = f2
    return front


def hypervolume_2d(points, ref=REF):
    """Hiperobjętość 2D zbioru punktów względem punktu odniesienia ``ref``."""
    if not points:
        return 0.0
    pts = sorted(points, key=lambda p: (p[0], -p[1]))
    hv, prev_f2 = 0.0, ref[1]
    for f1, f2 in pts:
        w, h = ref[0] - f1, prev_f2 - f2
        if w > 0 and h > 0:
            hv += w * h
        prev_f2 = f2
    return hv


def load_global_norms():
    """Wczytaj zamrożone globalne granice min/max (per instancja) z pracy."""
    path = REPO / "results" / "global_norms.json"
    return json.loads(path.read_text())["norms"]


def hv_normed_global(archive, norms_inst):
    """HV archiwum znormalizowanego globalnymi granicami danej instancji."""
    i0, i1 = norms_inst["min_f1"], norms_inst["min_f2"]
    n0, n1 = norms_inst["max_f1"], norms_inst["max_f2"]
    r0, r1 = (n0 - i0) or 1.0, (n1 - i1) or 1.0
    norm = [((f1 - i0) / r0, (f2 - i1) / r1) for f1, f2 in archive]
    return hypervolume_2d(non_dominated(norm), REF)
