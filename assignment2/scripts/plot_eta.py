"""Plot DLA clusters grown at different η values side-by-side."""

import numpy as np
import matplotlib.pyplot as plt
import os, glob as globmod

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
})

eta_values = [0.0, 0.5, 1.0, 2.0, 3.0]

# ── Load cluster masks ───────────────────────────────────────────────────────
clusters = []
loaded_etas = []
for ev in eta_values:
    tag = str(ev)
    if tag.endswith(".0"):
        tag = tag[:-2]
    # Try multiple filename formats.
    for candidate in [f"output/dla_cluster_eta_{tag}.txt",
                      f"output/dla_cluster_eta_{ev}.txt"]:
        if os.path.exists(candidate):
            clusters.append(np.loadtxt(candidate, dtype=int))
            loaded_etas.append(ev)
            break

if not clusters:
    raise FileNotFoundError("No η cluster files found in output/")

n = len(clusters)

# ── 1. Side-by-side cluster masks ────────────────────────────────────────────
fig, axes = plt.subplots(1, n, figsize=(4 * n, 4.5), squeeze=False)
axes = axes[0]
for ax, mask, ev in zip(axes, clusters, loaded_etas):
    ax.imshow(mask, cmap="binary", origin="lower", extent=[0, 1, 0, 1])
    n_occ = int(mask.sum())
    ax.set_title(f"η = {ev}\n({n_occ} cells)")
    ax.set_xlabel("x")
    ax.set_aspect("equal")
axes[0].set_ylabel("y")
fig.suptitle("DLA Clusters at Different η Values", fontsize=16, y=1.02)
fig.tight_layout()
fig.savefig("output/plots/eta_clusters.png", dpi=150, bbox_inches="tight")
print("saved output/plots/eta_clusters.png")
plt.close("all")

# ── 2. Load and plot concentration fields if available ────────────────────────
concs = []
conc_etas = []
for ev in loaded_etas:
    tag = str(ev)
    if tag.endswith(".0"):
        tag = tag[:-2]
    for candidate in [f"output/dla_conc_eta_{tag}.txt",
                      f"output/dla_conc_eta_{ev}.txt"]:
        if os.path.exists(candidate):
            concs.append(np.loadtxt(candidate))
            conc_etas.append(ev)
            break

if concs:
    fig2, axes2 = plt.subplots(1, len(concs), figsize=(4 * len(concs), 4.5),
                               squeeze=False)
    axes2 = axes2[0]
    for ax, conc, ev in zip(axes2, concs, conc_etas):
        im = ax.imshow(conc, cmap="viridis", origin="lower",
                       extent=[0, 1, 0, 1], vmin=0, vmax=1)
        ax.set_title(f"η = {ev}")
        ax.set_xlabel("x")
        ax.set_aspect("equal")
    axes2[0].set_ylabel("y")
    fig2.colorbar(im, ax=axes2.tolist(), label=r"$c(x,y)$", shrink=0.8)
    fig2.suptitle("Concentration Fields at Different η Values",
                  fontsize=16, y=1.02)
    fig2.tight_layout()
    fig2.savefig("output/plots/eta_concentrations.png", dpi=150,
                 bbox_inches="tight")
    print("saved output/plots/eta_concentrations.png")
    plt.close("all")
