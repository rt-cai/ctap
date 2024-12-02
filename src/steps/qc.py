import os, sys
from pathlib import Path
import scanpy as sc
import anndata as ad
import warnings

warnings.filterwarnings("ignore")
_, adata_path = sys.argv
adata_path = Path(adata_path)
adata = ad.read(adata_path)

# ------------------------------------------
# view data
sc.pp.calculate_qc_metrics(
    adata, inplace=True, log1p=True
)
ax = sc.pl.violin(
    adata,
    ["n_genes_by_counts", "total_counts", "pct_counts_in_top_100_genes"],
    jitter=0.4,
    multi_panel=True,
    show=False,
)
ax.figure.savefig("qc_metrics.png")
