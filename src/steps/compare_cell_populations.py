import os, sys
from pathlib import Path
import anndata as ad
import numpy as np
import warnings

from local.figures.template import BaseFigure, ApplyTemplate, go
from local.figures.colors import Color, COLORS, Palettes
import plotly.io as pio

pio.kaleido.scope.chromium_args += (
    "--disable-gpu",
    "--single-process",
)
warnings.filterwarnings("ignore")
_, adata_path = sys.argv
adata_path = Path(adata_path)
adata = ad.read(adata_path)

# ------------------------------------------
# get counts per cell type
all_counts = {}
for sample in adata.obs["sample_type"].unique():
    _df = adata.obs
    _df = _df[_df["sample_type"] == sample]
    counts = _df[["cell_type"]].value_counts()
    meta = all_counts.get(sample, {})
    for k, v in counts.items():
        meta[k[0]] = v
    all_counts[sample] = meta

# ------------------------------------------
# compare relative counts for infected and control
colors = {
    "Control": Palettes.PLOTLY[0],
    "Infected": Palettes.PLOTLY[1],
}
fig = BaseFigure()
c = np.array([v for data in all_counts.values() for v in data.values()]).sum()
m = 0
for sample, data in all_counts.items():
    a = -1 if "Infected" in sample else 1
    fig.add_trace(go.Bar(
        y=list(data.keys()),
        x=[a*v/c for v in data.values()],
        name=sample,
        marker = dict(
            color=colors[sample].color_value,
            line_width=0,
        ),
        orientation='h',
    ))
    m = max(m, max(data.values())/c)
rx = m*1.1
fig = ApplyTemplate(
    fig,
    default_xaxis=dict(dtick=0.2, range=[-rx, rx], title="Fraction of Cells"),
    default_yaxis=dict(title="Cell Type"),
    layout=dict(
        width=800, height=350,
        barmode='relative',
        font_size=14,
    ),
)
fig.write_image("compare_cell_populations.png")
