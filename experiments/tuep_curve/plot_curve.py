"""Plot the TUEP learning curve (AUC and balanced accuracy vs training patients) in the NeuroState deck style."""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

res = Path(sys.argv[1]); out_png = Path(sys.argv[2])
S = json.loads((res / "curve_summary.json").read_text())
rows = list(csv.DictReader(open(res / "curve_folds.csv")))
sizes = [int(k) for k in S["curve"].keys()]
BG, CARD, TEXT, MUTED, GOLD, BLUE, GRID = "#121A21", "#1A242D", "#ECE6DA", "#A9B3BC", "#F0B45A", "#5B8FD9", "#2C3640"
plt.rcParams.update({"font.family": "Segoe UI", "font.size": 12})
fig, ax = plt.subplots(figsize=(8.6, 4.6), dpi=220)
fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
for key, col, lab in (("auc", GOLD, "AUC"), ("bacc", BLUE, "Balanced accuracy")):
    m = np.array([S["curve"][str(n)][f"{key}_mean"] for n in sizes])
    se = np.array([S["curve"][str(n)][f"{key}_se"] for n in sizes])
    ax.fill_between(sizes, m - 1.96 * se, m + 1.96 * se, color=col, alpha=0.16, lw=0)
    ax.plot(sizes, m, color=col, lw=2.4, marker="o", ms=8, mec=BG, mew=2, label=lab)
    for x, v in zip(sizes, m):
        ax.annotate(f"{v:.2f}", (x, v), textcoords="offset points", xytext=(0, 11 if key == "auc" else -19),
                    ha="center", color=col, fontsize=11, fontweight="bold")
ax.axhline(0.5, color=MUTED, lw=1, ls=(0, (4, 4)))
ax.text(sizes[-1], 0.505, "chance", color=MUTED, fontsize=10, ha="right", va="bottom")
ax.set_xscale("log", base=2); ax.set_xticks(sizes, [str(s) for s in sizes])
ax.set_xlabel("Patients used for training (half epilepsy, half no epilepsy)", color=MUTED)
ax.set_ylabel("Score on 40 held-out patients", color=MUTED)
lo = min(0.45, min(S["curve"][str(n)]["bacc_mean"] for n in sizes) - 0.08)
ax.set_ylim(lo, max(0.85, max(S["curve"][str(n)]["auc_mean"] for n in sizes) + 0.08))
for s in ("top", "right"): ax.spines[s].set_visible(False)
for s in ("left", "bottom"): ax.spines[s].set_color(GRID)
ax.tick_params(colors=MUTED); ax.grid(axis="y", color=GRID, lw=0.7); ax.set_axisbelow(True)
leg = ax.legend(loc="upper left", frameon=False, labelcolor=TEXT)
fig.tight_layout()
fig.savefig(out_png, facecolor=BG)
print("saved", out_png)
