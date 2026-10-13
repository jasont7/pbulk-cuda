"""Shared look of the report figures. Colours are a validated two-hue pair plus greys."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCATTER = "#eb6834"    # the scatter kernel, everywhere
GATHER = "#2a78d6"     # the gather kernel, everywhere
CPU = "#52514e"        # CPU baselines
INK = "#0b0b0b"
MUTED = "#6b6a66"
GRID = "#dddcd7"
TINT_A = "#eef3fa"     # donor A band
TINT_B = "#f6f1e8"     # donor B band

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 7,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.5, "xtick.color": MUTED, "ytick.color": INK,
    "text.color": INK, "axes.labelcolor": INK, "pdf.fonttype": 42, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})
