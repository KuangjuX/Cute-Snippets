import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for clean PDF/PNG output
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches

# ============== Academic Paper Style Configuration ==============
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 8,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6.5,
        "figure.dpi": 100,  # Screen DPI for figure creation
        "savefig.dpi": 300,  # High DPI only for raster output (PNG)
        "axes.linewidth": 0.5,
        "grid.linewidth": 0.3,
        "lines.linewidth": 0.6,
        "patch.linewidth": 0.3,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,  # TrueType fonts in PDF for compatibility
    }
)

# ============== Color Palette (Academic/Professional) ==============
colors = {
    "Cute-Snippets": "#2E86AB",  # Strong blue (our system - prominent)
    "torch.compile": "#F18F01",  # Orange
    "Liger Kernel": "#C73E1D",  # Brick red
}

# Hatching patterns for print-friendly distinction
hatches = {
    "Cute-Snippets": "",
    "torch.compile": "\\\\\\",
    "Liger Kernel": "...",
}

# ============== Shared x-axis labels ==============
# 7 representative [M, N] configurations covering small to very large
sizes = ["32K×1K", "32K×4K", "8K×8K", "16K×16K", "4K×32K", "4K×64K", "4K×128K"]

frameworks = ["Cute-Snippets", "torch.compile", "Liger Kernel"]

# ============== Softmax Data ==============
softmax_latency = {
    "Cute-Snippets": [0.0485, 0.1833, 0.0939, 0.3622, 0.1822, 0.3618, 0.7169],
    "torch.compile": [0.0588, 0.3942, 0.1923, 0.7030, 0.3635, 0.7451, 1.4992],
    "Liger Kernel": [0.0479, 0.1804, 0.0909, 0.3597, 0.1908, 0.5815, None],
}
softmax_bandwidth = {
    "Cute-Snippets": [2765, 2930, 2858, 2964, 2946, 2968, 2996],
    "torch.compile": [2283, 1362, 1396, 1527, 1477, 1441, 1432],
    "Liger Kernel": [2799, 2976, 2954, 2985, 2814, 1847, None],
}

# ============== RMSNorm Data ==============
# Same 7 representative [M, N] configurations
rmsnorm_latency = {
    "Cute-Snippets": [0.0489, 0.1833, 0.0944, 0.3627, 0.1844, 0.3645, 0.7234],
    "torch.compile": [0.0657, 0.2227, 0.1262, 0.4595, 0.2940, 0.6268, 1.2647],
    "Liger Kernel": [0.0589, 0.1830, 0.0912, 0.3611, 0.1827, 1.1735, None],
}
rmsnorm_bandwidth = {
    "Cute-Snippets": [2743, 2928, 2843, 2960, 2911, 2946, 2969],
    "torch.compile": [2043, 2411, 2128, 2337, 1827, 1713, 1698],
    "Liger Kernel": [2280, 2934, 2942, 2973, 2939, 915, None],
}


# ============== Drawing Functions ==============
def draw_latency_bar(
    ax, x_labels, fws, times, title, xlabel, show_ylabel=True, log_scale=True
):
    """Draw a grouped bar chart for latency with academic styling."""
    n_groups = len(x_labels)
    n_bars = len(fws)
    bar_width_ratio = 0.75
    bar_width = bar_width_ratio / n_bars
    positions = np.arange(n_groups)

    for i, fw in enumerate(fws):
        offset = (i - n_bars / 2 + 0.5) * bar_width
        values = []
        valid_pos = []
        for j in range(n_groups):
            v = times[fw][j]
            if v is not None:
                values.append(v)
                valid_pos.append(positions[j] + offset)
        ax.bar(
            valid_pos,
            values,
            bar_width * 0.9,
            label=fw,
            color=colors[fw],
            edgecolor="black",
            linewidth=0.3,
            hatch=hatches[fw],
            zorder=3,
        )

    # Styling
    ax.set_title(title, fontweight="bold", pad=4, fontsize=7)
    ax.set_xlabel(xlabel, labelpad=2, fontsize=6)
    if show_ylabel:
        ax.set_ylabel("Latency (ms)", labelpad=1, fontsize=6)

    ax.set_xticks(positions)
    ax.set_xticklabels(x_labels, rotation=30, ha="right", fontsize=5.5)

    if log_scale:
        ax.set_yscale("log")

    # Mark unsupported — MUST be after set_yscale('log')
    for i, fw in enumerate(fws):
        offset = (i - n_bars / 2 + 0.5) * bar_width
        for j in range(n_groups):
            if times[fw][j] is None:
                ylim = ax.get_ylim()
                y_pos = ylim[0] * 1.2 if log_scale else 0.01
                ax.text(
                    positions[j] + offset,
                    y_pos,
                    "N/A",
                    ha="center",
                    va="bottom",
                    fontsize=5.5,
                    rotation=90,
                    color=colors[fw],
                    fontweight="bold",
                    zorder=5,
                )

    ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0, linewidth=0.3)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.4)
    ax.spines["bottom"].set_linewidth(0.4)
    ax.tick_params(
        axis="both", which="major", labelsize=5.5, width=0.4, length=2
    )
    return ax


def draw_bandwidth_bar(
    ax, x_labels, fws, bw_data, title, xlabel, show_ylabel=True
):
    """Draw a grouped bar chart for bandwidth (GB/s) — higher is better."""
    n_groups = len(x_labels)
    n_bars = len(fws)
    bar_width_ratio = 0.75
    bar_width = bar_width_ratio / n_bars
    positions = np.arange(n_groups)

    for i, fw in enumerate(fws):
        offset = (i - n_bars / 2 + 0.5) * bar_width
        values = []
        valid_pos = []
        for j in range(n_groups):
            v = bw_data[fw][j]
            if v is not None:
                values.append(v)
                valid_pos.append(positions[j] + offset)
        ax.bar(
            valid_pos,
            values,
            bar_width * 0.9,
            label=fw,
            color=colors[fw],
            edgecolor="black",
            linewidth=0.3,
            hatch=hatches[fw],
            zorder=3,
        )

    # Mark unsupported
    for i, fw in enumerate(fws):
        offset = (i - n_bars / 2 + 0.5) * bar_width
        for j in range(n_groups):
            if bw_data[fw][j] is None:
                ax.text(
                    positions[j] + offset,
                    50,
                    "N/A",
                    ha="center",
                    va="bottom",
                    fontsize=5.5,
                    rotation=90,
                    color=colors[fw],
                    fontweight="bold",
                    zorder=5,
                )

    # H800 peak bandwidth reference line
    h800_peak_bw = 3350  # GB/s (H800 HBM3 peak)
    ax.axhline(
        y=h800_peak_bw, color="#888888", linestyle=":", linewidth=0.6, zorder=2
    )
    ax.text(
        n_groups - 0.5,
        h800_peak_bw + 50,
        "H800 Peak",
        ha="right",
        va="bottom",
        fontsize=4.5,
        color="#666666",
        style="italic",
    )

    # Styling
    ax.set_title(title, fontweight="bold", pad=4, fontsize=7)
    ax.set_xlabel(xlabel, labelpad=2, fontsize=6)
    if show_ylabel:
        ax.set_ylabel("Bandwidth (GB/s)", labelpad=1, fontsize=6)

    ax.set_xticks(positions)
    ax.set_xticklabels(x_labels, rotation=30, ha="right", fontsize=5.5)

    ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0, linewidth=0.3)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.4)
    ax.spines["bottom"].set_linewidth(0.4)
    ax.tick_params(
        axis="both", which="major", labelsize=5.5, width=0.4, length=2
    )
    ax.set_ylim(0, 3700)
    return ax


# ============== Create Combined Figure ==============
# Layout: 2 rows × 3 columns (last column = shared legend)
# Row 1: (a) Softmax Latency, (b) Softmax Bandwidth
# Row 2: (c) RMSNorm Latency, (d) RMSNorm Bandwidth
fig = plt.figure(figsize=(5.5, 3.6))

gs = GridSpec(
    2,
    3,
    figure=fig,
    width_ratios=[1, 1, 0.28],
    hspace=0.65,
    wspace=0.38,
    left=0.08,
    right=0.98,
    top=0.93,
    bottom=0.10,
)

labels = ["(a)", "(b)", "(c)", "(d)"]

# ---- Row 1: Softmax ----
ax_a = fig.add_subplot(gs[0, 0])
draw_latency_bar(
    ax_a,
    sizes,
    frameworks,
    softmax_latency,
    "Softmax Latency (H800)",
    "[M, N]",
    show_ylabel=True,
)
ax_a.text(
    -0.18,
    1.02,
    labels[0],
    transform=ax_a.transAxes,
    fontsize=8,
    fontweight="bold",
    va="bottom",
)

ax_b = fig.add_subplot(gs[0, 1])
draw_bandwidth_bar(
    ax_b,
    sizes,
    frameworks,
    softmax_bandwidth,
    "Softmax Bandwidth (H800)",
    "[M, N]",
    show_ylabel=True,
)
ax_b.text(
    -0.18,
    1.02,
    labels[1],
    transform=ax_b.transAxes,
    fontsize=8,
    fontweight="bold",
    va="bottom",
)

# ---- Row 2: RMSNorm ----
ax_c = fig.add_subplot(gs[1, 0])
draw_latency_bar(
    ax_c,
    sizes,
    frameworks,
    rmsnorm_latency,
    "RMSNorm Latency (H800)",
    "[M, N]",
    show_ylabel=True,
)
ax_c.text(
    -0.18,
    1.02,
    labels[2],
    transform=ax_c.transAxes,
    fontsize=8,
    fontweight="bold",
    va="bottom",
)

ax_d = fig.add_subplot(gs[1, 1])
draw_bandwidth_bar(
    ax_d,
    sizes,
    frameworks,
    rmsnorm_bandwidth,
    "RMSNorm Bandwidth (H800)",
    "[M, N]",
    show_ylabel=True,
)
ax_d.text(
    -0.18,
    1.02,
    labels[3],
    transform=ax_d.transAxes,
    fontsize=8,
    fontweight="bold",
    va="bottom",
)

# ---- Shared Legend (spanning both rows in column 3) ----
legend_ax = fig.add_subplot(gs[:, 2])
legend_ax.axis("off")

all_frameworks = ["Cute-Snippets", "torch.compile", "Liger Kernel"]
handles = []
for fw in all_frameworks:
    patch = mpatches.Patch(
        facecolor=colors[fw],
        edgecolor="black",
        linewidth=0.4,
        hatch=hatches[fw],
        label=fw,
    )
    handles.append(patch)

legend = legend_ax.legend(
    handles=handles,
    loc="center",
    ncol=1,
    frameon=True,
    fancybox=False,
    edgecolor="#333333",
    fontsize=6.5,
    handlelength=1.5,
    handleheight=1.0,
    columnspacing=0.8,
    handletextpad=0.4,
    title="Frameworks",
    title_fontsize=7,
)
legend.get_frame().set_linewidth(0.5)

legend_ax.text(
    0.5,
    0.30,
    "(Cute-Snippets = Ours)",
    transform=legend_ax.transAxes,
    ha="center",
    va="center",
    fontsize=5.5,
    style="italic",
    color="#555555",
)
legend_ax.text(
    0.5,
    0.24,
    "N/A = unsupported",
    transform=legend_ax.transAxes,
    ha="center",
    va="center",
    fontsize=5.5,
    style="italic",
    color="#555555",
)

# ---- Save ----
fig.savefig(
    "media/kernels/evaluation.pdf",
    format="pdf",
    bbox_inches="tight",
    pad_inches=0.03,
)
fig.savefig(
    "media/kernels/evaluation.png",
    format="png",
    bbox_inches="tight",
    pad_inches=0.03,
    dpi=300,
)

print("Saved: evaluation.pdf and evaluation.png")
plt.close(fig)
