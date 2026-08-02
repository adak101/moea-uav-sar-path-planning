# Raport wyników — SMS-EMOA dla planowania ścieżek UAV

**Data:** 2026-05-23  
**Instancje:** 8 (tcB, tcT, tcV, tcW, tcGB, tcGL, tcLO, tcSW; węzłów: 259–599)  
**Seedów:** 30 | **Eval/run:** 30 000 | **UAV:** 3  
**Test statystyczny:** Wilcoxon signed-rank, α=0.05  
**Metryka:** HV znormalizowane globalnie per instancja, ref=(1.1, 1.1)

---

## Faza 0 — Inicjalizacja hybrydowa vs losowa

**Parametry:** pop=100, pc=0.6, pm=1.0, wagi uniform  
**Pary:** 240 (30 seedów × 8 instancji)

### Część 1 — Jakość populacji startowej (t=0)

| Instancja | Hybryda avg | Random avg | p | Lepszy |
|---|---|---|---|---|
| tcB | 0.833 | 0.571 | <0.001 | Hybryda |
| tcT | 0.662 | 0.262 | <0.001 | Hybryda |
| tcV | 0.608 | 0.232 | <0.001 | Hybryda |
| tcW | 1.111 | 0.687 | <0.001 | Hybryda |
| tcGB | 0.922 | 0.490 | <0.001 | Hybryda |
| tcGL | 0.597 | 0.138 | <0.001 | Hybryda |
| tcLO | 0.865 | 0.324 | <0.001 | Hybryda |
| tcSW | 0.733 | 0.174 | <0.001 | Hybryda |

**Zbiorczy:** Hybryda=0.791, Random=0.360, Δ=**+119.96%**, W=0, p≈0  
**A=0.949** (efekt duży) → **TAK, Hybryda istotnie lepsza**

### Część 2 — Jakość wyników końcowych (po 30 000 eval)

**Zbiorczy:** Hybryda≈0.930, Random≈0.237  
**A≈0.999** (efekt duży) → **TAK, Hybryda istotnie lepsza**

---

## Faza 1 — Ablacja 2×2

**Parametry:** pop=100, pc=0.6, pm=1.0, wagi uniform  
**Pary:** 240 per kontrast | **Korekcja:** Holm-Bonferroni (4 kontrasty)

### Wyniki per algorytm (średnie HV)

| Algorytm | Średnie HV |
|---|---|
| GEN-SMS-EMOA | 0.91007 |
| NSGA-II | 0.91173 |
| SMS-EMOA | 0.89744 |
| SS-NSGA-II | 0.89557 |

### Kontrasty ablacji 2×2

| Kontrast | Porównanie | p (Holm) | A | Efekt | Wniosek |
|---|---|---|---|---|---|
| A — kryterium @ steady-state | SMS-EMOA vs SS-NSGA-II | 0.355 | 0.506 | brak | brak różnicy |
| B — kryterium @ generacyjny | GEN-SMS-EMOA vs NSGA-II | 0.494 | 0.496 | brak | brak różnicy |
| C — architektura @ HV-contrib | SMS-EMOA vs GEN-SMS-EMOA | 0.013 | 0.438 | mały | GEN-SMS-EMOA lepszy |
| D — architektura @ crowding | SS-NSGA-II vs NSGA-II | <0.001 | 0.424 | mały | NSGA-II lepszy |

**Interpretacja:**
- Kryterium selekcji (HV-contribution vs crowding distance) **nie ma istotnego wpływu**
- Architektura (generacyjna vs steady-state) **ma mały, ale istotny wpływ** na korzyść generacyjnej
- Najlepszy konkurent dla SMS-EMOA: **NSGA-II** (0.912)

### LOO (Leave-One-Out) — stabilność wniosków

Wszystkie 4 kontrasty stabilne — żadna instancja nie zmienia decyzji (weryfikacja 8 × 4 = 32 podzbiorów).

---

## irace — Tuning hiperparametrów SMS-EMOA

**Budżet:** 5000 eksperymentów | **Parallel:** 4 | **Seed:** 2024  
**Instancje treningowe:** tcB, tcGB, tcGL, tcLO, tcV, tcW (6)  
**Instancje testowe (held-out):** tcT, tcSW  
**Czas:** ~4.35h (wall-clock)

### Najlepsza konfiguracja #534

| Parametr | Default | Tuned | Zmiana |
|---|---|---|---|
| w_invert | 0.167 | **0.821** | ↑ 4.9× |
| w_or_opt | 0.167 | 0.242 | ↑ 1.5× |
| w_transfer | 0.167 | 0.359 | ↑ 2.1× |
| w_swap | 0.167 | **0.042** | ↓ prawie wyłączony |
| w_cross | 0.167 | **0.748** | ↑ 4.5× |
| w_exchange | 0.167 | **0.051** | ↓ prawie wyłączony |
| pop | 100 | 108 | ≈ podobny |
| pc | 0.6 | **0.279** | ↓ rzadsze krzyżowanie |
| pm | 1.0 | 0.939 | ≈ podobny |

**Kluczowe odkrycia:**
- Operatory geometryczne dominują: **invert (2-opt intra) + cross_route (2-opt* inter)**
- Proste zamiany zostały prawie wyłączone: **swap ≈ 0, exchange ≈ 0**
- Krzyżowanie powinno być rzadsze (pc≈0.28), mutacja prawie zawsze (pm≈0.94)
- Wzorzec spójny we wszystkich 5 elitach

---

## Faza 2 — Efekt tuningu

**Default:** phase1_v2, pop=100, pc=0.6, pm=1.0, wagi uniform  
**Tuned:** phase2_tuned, pop=108, pc=0.2786, pm=0.9388, wagi irace #534  
**Pary:** 240

| Instancja | Default avg | Tuned avg | p | Lepszy |
|---|---|---|---|---|
| tcB | 0.936 | 0.981 | <0.001 | Tuned |
| tcT | 0.915 | 0.915 | 0.952 | ≈ remis |
| tcV | 0.955 | 0.953 | 0.824 | ≈ remis |
| tcW | 0.590 | 0.646 | 0.177 | Tuned (n.s.) |
| tcGB | 0.882 | 0.974 | <0.001 | Tuned |
| tcGL | 0.838 | 0.848 | 0.043 | Tuned |
| tcLO | 0.931 | 0.960 | <0.001 | Tuned |
| tcSW | 0.974 | 0.993 | <0.001 | Tuned |

**Zbiorczy:** Default=0.877, Tuned=0.909, Δ=**+3.55%**  
W=6259, p≈0, **A=0.649** (efekt średni) → **TAK, Tuned istotnie lepszy**

---

## Faza 3 — Porównanie końcowe

**SMS-EMOA tuned** (irace #534) vs **NSGA-II** (najlepszy z Fazy 1, parametry default)  
**Pary:** 240

| Instancja | NSGA-II avg | Tuned avg | p | Lepszy |
|---|---|---|---|---|
| tcB | 0.949 | 0.983 | <0.001 | Tuned |
| tcT | 0.911 | 0.907 | 0.516 | ≈ remis |
| tcV | 0.962 | 0.955 | 0.064 | ≈ remis |
| tcW | 0.650 | 0.681 | 0.503 | Tuned (n.s.) |
| tcGB | 0.922 | 0.974 | 0.002 | Tuned |
| tcGL | 0.850 | 0.852 | 0.761 | ≈ remis |
| tcLO | 0.962 | 0.974 | 0.158 | Tuned (n.s.) |
| tcSW | 0.978 | 0.991 | 0.004 | Tuned |

**Zbiorczy:** NSGA-II=0.898, SMS-EMOA tuned=0.914, Δ=**+1.81%**  
W=10227, p=0.000084, **A=0.592** (efekt mały) → **TAK, Tuned istotnie lepszy**

---

## Podsumowanie całości

| Faza | Porównanie | Δ HV | A | Efekt | Istotne |
|---|---|---|---|---|---|
| 0 — Inicjalizacja | Hybryda vs Random (t=0) | +120% | 0.949 | duży | TAK |
| 0 — Inicjalizacja | Hybryda vs Random (końcowe) | >>100% | ~0.999 | duży | TAK |
| 1 — Ablacja (kryterium) | SMS-EMOA vs SS-NSGA-II | +0.2% | 0.506 | brak | NIE |
| 1 — Ablacja (kryterium) | GEN-SMS-EMOA vs NSGA-II | -0.2% | 0.496 | brak | NIE |
| 1 — Ablacja (architektura) | GEN-SMS-EMOA vs SMS-EMOA | +1.4% | 0.438* | mały | TAK |
| 1 — Ablacja (architektura) | NSGA-II vs SS-NSGA-II | +1.8% | 0.424* | mały | TAK |
| 2 — Tuning | SMS-EMOA tuned vs default | +3.6% | 0.649 | średni | TAK |
| 3 — Finalne | SMS-EMOA tuned vs NSGA-II | +1.8% | 0.592 | mały | TAK |

*A < 0.5 oznacza że drugi algorytm w porównaniu wygrywa

### Główne wnioski

1. **Inicjalizacja hybrydowa jest kluczowa** — zapewnia +120% HV na starcie i utrzymuje przewagę do końca
2. **Kryterium selekcji nie ma znaczenia** — HV-contribution i crowding distance dają statystycznie identyczne wyniki
3. **Architektura generacyjna ma małą przewagę** nad steady-state przy parametrach domyślnych
4. **Tuning irace istotnie poprawia SMS-EMOA** — efekt średni (+3.6%), operatory geometryczne (invert, cross_route) dominują
5. **SMS-EMOA tuned przewyższa NSGA-II** — efekt mały (+1.8%), przewaga widoczna na 6/8 instancji
