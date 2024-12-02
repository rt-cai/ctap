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
# Normalize the data
adata.layers['raw_counts'] = adata.X.copy()
sc.pp.log1p(adata)
adata.layers['log_norm'] = adata.X.copy()

out_path = f"adata.{Path(__file__).stem}.h5ad"
adata.write(out_path)
