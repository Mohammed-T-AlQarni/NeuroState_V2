"""Chart for the stronger TUEP test: AUC vs training patients, one line per method; panel 2 = positive control (age)."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

res = Path(sys.argv[1]); out_png = Path(sys.argv[2])
BG, TEXT, MUTED, GRID = "#121A21", "#ECE6DA", "#A9B3BC", "#2C3640"
COL = {"eegnet": "#F0B45A", "microstates": "#5B8FD9", "spectral": "#7FC8A9"}
NAME = {"eegnet": "EEGNet (raw EEG, GPU)", "microstates": "Microstates (24 parameters)", "spectral": "Spectral band power"}
plt.rcParams.update({"font.family": "Segoe UI", "font.size": 11})
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), dpi=200, sharey=True)
fig.patch.set_facecolor(BG)
for ax, target, title in zip(axes, ("epilepsy", "age"), ("Epilepsy vs no epilepsy", "Positive control: older vs younger")):
    ax.set_facecolor(BG)
    xs_all, placed = None, []
    for m in ("eegnet", "microstates", "spectral"):
        f = res / f"summary_{m}_{target}.json"
        if not f.exists():
            continue
        J = json.loads(f.read_text()); S = J["curve"]
        keys = list(S.keys())
        nall = round(0.8 * J["patients"])
        xs = [int(k) if k != "all" else nall for k in keys]
        xs_all = xs
        mu = [S[k]["auc_mean"] for k in keys]; se = [S[k]["auc_se"] for k in keys]
        ax.fill_between(xs, [a - 1.96 * b for a, b in zip(mu, se)], [a + 1.96 * b for a, b in zip(mu, se)], color=COL[m], alpha=0.15, lw=0)
        ax.plot(xs, mu, color=COL[m], lw=2.2, marker="o", ms=6, mec=BG, mew=1.5, label=NAME[m])
        ly = mu[-1]
        while any(abs(ly - p) < 0.022 for p in placed):
            ly += 0.024
        placed.append(ly)
        ax.annotate(f"{mu[-1]:.2f}", (xs[-1], ly), textcoords="offset points", xytext=(8, -4), color=COL[m], fontsize=10,
                    fontweight="bold", annotation_clip=False)
    ax.axhline(0.5, color=MUTED, lw=1, ls=(0, (4, 4)))
    ax.text(28, 0.493, "chance", color=MUTED, fontsize=9, va="top")
    if xs_all:
        ax.set_xscale("log", base=2)
        ax.set_xticks(xs_all, [str(x) for x in xs_all[:-1]] + [f"all ({xs_all[-1]})"])
    ax.set_title(title, color=TEXT, fontsize=12, loc="left")
    ax.set_xlabel("Patients used for training", color=MUTED)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED); ax.grid(axis="y", color=GRID, lw=0.7); ax.set_axisbelow(True)
axes[0].set_ylabel("AUC on held-out patients", color=MUTED)
axes[0].set_ylim(0.4, 0.9)
axes[0].legend(loc="upper left", frameon=False, labelcolor=TEXT, fontsize=9.5)
fig.tight_layout()
fig.savefig(out_png, facecolor=BG)
print("saved", out_png)
