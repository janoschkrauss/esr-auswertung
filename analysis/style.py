"""Figure style for the report: LaTeX font (Latin Modern), sizes and colors."""
import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

PREAMBLE = (r"\usepackage[T1]{fontenc}\usepackage{lmodern}\usepackage{amsmath}\usepackage{siunitx}"
            r"\sisetup{locale=US}")

plt.rcParams.update({
    "text.usetex": True, "text.latex.preamble": PREAMBLE, "font.family": "serif",
    "font.size": 8.5, "axes.labelsize": 8.5, "legend.fontsize": 7.5, "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
    "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 1.1, "lines.markersize": 3.2, "legend.frameon": False,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "figure.dpi": 150,
})

# Colors (colorblind-safe, printable)
BLUE, RED, GRAY, ORANGE = "#1f4e79", "#b03a2e", "#7f7f7f", "#d68910"
WIDTH = 6.3        # text width in inches (16 cm)


def tick(x, pos=None):
    """Tick label in math mode."""
    s = f"{x:g}"
    if "e" in s:
        s = f"{x:.6f}".rstrip("0").rstrip(".")
    return "$" + s + "$"


def math_ticks(*axes):
    for ax in axes:
        ax.xaxis.set_major_formatter(FuncFormatter(tick))
        ax.yaxis.set_major_formatter(FuncFormatter(tick))


def panel(ax, letter, x=-0.02, y=1.02):
    ax.text(x, y, rf"\textbf{{({letter})}}", transform=ax.transAxes, ha="right", va="bottom", fontsize=9)
