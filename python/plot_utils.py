"""Plotting only: numerical core never imports this module."""
from pathlib import Path
import io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIGURE_DIR = Path(__file__).resolve().parents[1] / "figures"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 12,
    "axes.labelsize": 12, "axes.titlesize": 12,
    "xtick.labelsize": 11, "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": False, "figure.figsize": (7.4, 3.7),
    "savefig.bbox": "tight", "pdf.fonttype": 42,
    "axes.prop_cycle": plt.cycler(color=[
        "#24577C", "#CE7334", "#469278", "#8E598C", "#5F6368"]),
})


def savefig(fig, name):
    """Save the same fully computed figure as vector PDF and preview PNG."""
    FIGURE_DIR.mkdir(exist_ok=True)
    fig.tight_layout()
    for extension in ("pdf", "png"):
        buffer = io.BytesIO()
        fig.savefig(buffer, format=extension, dpi=150)
        target = FIGURE_DIR / (name + "." + extension)
        temporary = target.with_suffix(".tmp")
        temporary.write_bytes(buffer.getvalue())
        temporary.replace(target)
    plt.close(fig)
