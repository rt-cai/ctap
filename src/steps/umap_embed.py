import os, sys
from pathlib import Path
import scanpy as sc
import anndata as ad
import decoupler as dc
import warnings

warnings.filterwarnings("ignore")
_, adata_path = sys.argv
adata_path = Path(adata_path)
adata = ad.read(adata_path)

# ------------------------------------------
# Generate PCA features
sc.tl.pca(adata, svd_solver='arpack')
# Restore X to be norm counts
dc.swap_layer(adata, 'log_norm', X_layer_key=None, inplace=True)
# Compute distances in the PCA space, and find cell neighbors
sc.pp.neighbors(adata, n_neighbors=12, n_pcs=40)
# Generate UMAP features
sc.tl.umap(
    adata,
    min_dist = 0.3,
    spread = 1,
)

# Visualize
ax = sc.pl.umap(adata, color='sample_type', s=5, alpha=0.5, title="HCMV Infection", show=False)
ax.figure.set_figwidth(6)
ax.figure.set_figheight(6)

ax.figure.savefig('umap_infection.png', dpi=300)
out_path = f"adata.{Path(__file__).stem}.h5ad"
adata.write(out_path)
