"""Consistent, quiet plotting style. Each component keeps one colour everywhere."""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

SURFACE = '#fcfcfb'
INK = '#0b0b0b'
INK_2 = '#52514e'
MUTED = '#898781'
GRID = '#e1e0d9'
AXIS = '#c3c2b7'

# validated categorical order (adjacent pairs CVD-safe): blue, orange, aqua, yellow
COLORS = {'always_on': '#2a78d6', 'heating': '#eb6834', 'other': '#1baf7a', 'ev': '#eda100'}
LABELS = {'always_on': 'Always-on', 'heating': 'Heating', 'other': 'Other', 'ev': 'EV charging'}
STACK = ('always_on', 'heating', 'other', 'ev')
TYPE_COLORS = {'neither': '#2a78d6', 'heating only': '#eb6834', 'EV only': '#eda100', 'EV + heating': '#6250d6'}


def setup():
    mpl.rcParams.update({
        'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
        'font.family': ['Segoe UI', 'DejaVu Sans'], 'font.size': 10,
        'text.color': INK, 'axes.labelcolor': INK_2, 'axes.titlecolor': INK,
        'axes.titlesize': 12, 'axes.titleweight': 'semibold', 'axes.titlelocation': 'left',
        'axes.edgecolor': AXIS, 'axes.linewidth': 0.8,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.8, 'axes.axisbelow': True,
        'xtick.color': MUTED, 'ytick.color': MUTED, 'xtick.labelcolor': INK_2, 'ytick.labelcolor': INK_2,
        'lines.linewidth': 2, 'lines.solid_capstyle': 'round',
        'legend.frameon': False, 'legend.fontsize': 9,
        'figure.dpi': 110, 'savefig.dpi': 150, 'savefig.bbox': 'tight',
    })


def title(ax, text, subtitle=None):
    """Left-aligned title with an optional subtitle in secondary ink, never overlapping."""
    if subtitle is None:
        ax.set_title(text, pad=10)
        return
    ax.set_title(text, pad=24)
    ax.annotate(subtitle, xy=(0, 1), xycoords='axes fraction', xytext=(0, 7), textcoords='offset points',
                color=INK_2, fontsize=9, va='bottom')


def stacked(ax, t, parts, total=None, legend=True, label_total='Measured total'):
    """Stacked areas with a thin surface gap between layers; measured total as a dark line."""
    ys = [parts[c] for c in STACK]
    ax.stackplot(t, *ys, colors=[COLORS[c] for c in STACK], labels=[LABELS[c] for c in STACK],
                 edgecolor=SURFACE, linewidth=0.8)
    if total is not None:
        ax.plot(t, total, color=INK, lw=1.2, label=label_total)
    ax.set_ylim(bottom=0)
    ax.set_ylabel('kWh per hour')
    if legend:
        ax.legend(loc='upper left', ncol=5, bbox_to_anchor=(0, 1.0))
    return ax


def style_dates(ax, fmt='%a %d %b'):
    import matplotlib.dates as mdates
    ax.xaxis.set_major_formatter(mdates.DateFormatter(fmt))
    ax.xaxis.set_major_locator(mdates.DayLocator())


def bar_end_labels(ax, bars, fmt='{:.0f}'):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), fmt.format(b.get_height()),
                ha='center', va='bottom', color=INK_2, fontsize=9)
