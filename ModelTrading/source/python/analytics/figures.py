"""Shared figure and table base for the thesis artefact pipeline.

Every artefact under ``docs/figures/`` and ``docs/tables/`` is written through this
module, so the whole set reads as one system: identical page widths, fonts, palette
and file naming. Each figure is written twice — a vector ``.pdf`` for
``includegraphics`` and a ``.png`` for quick review — and each table twice — ``.csv``
for re-analysis and ``.tex`` (booktabs) so numbers never get retyped into the thesis.

Palette
-------
The categorical hues are the validated default order. Re-checked with the data-viz
validator before adoption::

    node scripts/validate_palette.js "#2a78d6,#eb6834,#1baf7a,#eda100,#e87ba4,#008300" --mode light
    -> lightness band PASS, chroma floor PASS, adjacent CVD dE 9.1 PASS,
       normal-vision dE 19.6 PASS, contrast WARN on aqua/yellow/magenta

The contrast WARN obligates *relief*: every figure using those slots ships a legend
and, where a reader must map a mark to a name, a direct label. Scatter and other
forms that put all pairs on screen at once are capped at ``CATEGORICAL_ALL_PAIRS``
(the first three slots), which validates under ``--pairs all`` (worst CVD dE 9.2,
normal-vision dE 24.0).

Rules that hold for every figure here (the ones that go wrong most often): never a
second y-axis, sequential encoding is one hue light-to-dark, diverging is two hues
with a neutral grey midpoint, grid and axes stay recessive, and a legend is present
whenever more than one series shares a panel.
"""

import json
import os

import matplotlib
matplotlib.use('Agg')          # figures are files, never windows
import matplotlib.pyplot as plt

# --- Palette -----------------------------------------------------------------------
# Categorical slots, in fixed order. Assign by position, never cycle: a series keeps
# its colour when a filter removes its neighbours.
CATEGORICAL = ('#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300',
               '#4a3aa7', '#e34948')
# Forms that show every pair at once (scatter, bubble, small multiples) cap here.
CATEGORICAL_ALL_PAIRS = CATEGORICAL[:3]

# Sequential: one hue, light to dark. For heatmaps of a magnitude.
SEQUENTIAL = ('#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b')
# Diverging: warm/cool poles with a neutral grey midpoint that reads as "nothing".
DIVERGING_LOW, DIVERGING_MID, DIVERGING_HIGH = '#2a78d6', '#f0efec', '#e34948'

# Status colours are reserved and never stand in for a series.
STATUS = {'good': '#0ca30c', 'warning': '#fab219', 'serious': '#ec835a', 'critical': '#d03b3b'}

# Chrome and ink.
SURFACE = '#fcfcfb'
INK_PRIMARY = '#0b0b0b'
INK_SECONDARY = '#52514e'
INK_MUTED = '#898781'
GRIDLINE = '#e1e0d9'
BASELINE = '#c3c2b7'

# The four models always get the same slot, in every figure of the thesis.
MODEL_COLORS = {
    'long_slow': CATEGORICAL[0],
    'short_slow': CATEGORICAL[1],
    'long_fast': CATEGORICAL[2],
    'short_fast': CATEGORICAL[3],
}

# --- Page geometry -----------------------------------------------------------------
# A4 with 2.5 cm margins gives a ~16 cm text block.
WIDTH_FULL = 6.3      # inches, spans the text block
WIDTH_HALF = 3.05     # two side by side
_GOLDEN = 0.618


def size(width=WIDTH_FULL, ratio=_GOLDEN):
    """Figure size in inches for a given text width and height:width ratio."""
    return (width, width * ratio)


def apply_style():
    """Install the print rcParams. Idempotent; called by every entry point."""
    plt.rcParams.update({
        'figure.facecolor': SURFACE,
        'axes.facecolor': SURFACE,
        'savefig.facecolor': SURFACE,
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Segoe UI', 'Arial'],
        'font.size': 8,
        'axes.titlesize': 9,
        'axes.labelsize': 8,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'legend.fontsize': 7,
        'axes.edgecolor': BASELINE,
        'axes.labelcolor': INK_SECONDARY,
        'text.color': INK_PRIMARY,
        'xtick.color': INK_MUTED,
        'ytick.color': INK_MUTED,
        'grid.color': GRIDLINE,
        'grid.linewidth': 0.6,
        'axes.linewidth': 0.8,
        'lines.linewidth': 1.4,
        'lines.markersize': 4,
        'legend.frameon': False,
        'figure.dpi': 110,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'pdf.fonttype': 42,      # embed TrueType, so the PDF keeps selectable text
        'ps.fonttype': 42,
    })


def tidy(ax, grid_axis='y'):
    """Recede the chrome: drop the top/right spines, put a hairline grid behind."""
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(BASELINE)
    if grid_axis:
        ax.grid(True, axis=grid_axis, linewidth=0.6, color=GRIDLINE, zorder=0)
        ax.set_axisbelow(True)
    return ax


def model_color(model_key, fallback_index=0):
    """Stable colour for one of the four models; falls back to a categorical slot."""
    return MODEL_COLORS.get(model_key, CATEGORICAL[fallback_index % len(CATEGORICAL)])


class ArtefactWriter:
    """Writes one stage's figures and tables, and records what it wrote.

    The record is what the pipeline manifest consumes, so a figure referenced in the
    thesis text can always be traced back to the stage and the command that made it.

    Args:
        stage: stage id, e.g. ``'S7'``. Becomes the subdirectory under figures/ and
            tables/ and the prefix of the LaTeX label.
        root: the ``docs/`` directory.
    """

    def __init__(self, stage, root):
        self.stage = stage
        self.root = os.path.abspath(root)
        self.figure_dir = os.path.join(self.root, 'figures', stage)
        self.table_dir = os.path.join(self.root, 'tables', stage)
        self.artifacts = []
        apply_style()

    @staticmethod
    def _ensure(path):
        os.makedirs(path, exist_ok=True)
        return path

    def save_figure(self, fig, name, caption=None):
        """Write a figure as PDF (for LaTeX) and PNG (for review). Closes the figure."""
        self._ensure(self.figure_dir)
        paths = {}
        for ext in ('pdf', 'png'):
            p = os.path.join(self.figure_dir, name + '.' + ext)
            fig.savefig(p)
            paths[ext] = p
        plt.close(fig)
        self.artifacts.append({'kind': 'figure', 'stage': self.stage, 'name': name,
                               'caption': caption, 'paths': paths})
        return paths

    def save_table(self, df, name, caption=None, float_format='%.4f', index=False):
        """Write a table as CSV (re-analysis) and LaTeX booktabs (typesetting)."""
        self._ensure(self.table_dir)
        csv_path = os.path.join(self.table_dir, name + '.csv')
        tex_path = os.path.join(self.table_dir, name + '.tex')
        df.to_csv(csv_path, index=index)
        kwargs = dict(index=index, escape=True, float_format=float_format)
        try:
            tex = df.to_latex(caption=caption, label='tab:' + self.stage + '_' + name,
                              **kwargs)
        except TypeError:      # pandas too old for caption/label
            tex = df.to_latex(**kwargs)
        with open(tex_path, 'w', encoding='utf-8') as fh:
            fh.write(tex)
        paths = {'csv': csv_path, 'tex': tex_path}
        self.artifacts.append({'kind': 'table', 'stage': self.stage, 'name': name,
                               'caption': caption, 'paths': paths, 'n_rows': int(len(df))})
        return paths

    def save_json(self, obj, name):
        """Write a machine-readable result under ``docs/results/<stage>/``."""
        d = self._ensure(os.path.join(self.root, 'results', self.stage))
        path = os.path.join(d, name + '.json')
        with open(path, 'w', encoding='utf-8') as fh:
            json.dump(obj, fh, indent=2, default=str)
        self.artifacts.append({'kind': 'result', 'stage': self.stage, 'name': name,
                               'paths': {'json': path}})
        return path

    def relative(self, path):
        """Path relative to the docs root — what goes into the markdown reports."""
        return os.path.relpath(path, self.root).replace(os.sep, '/')
