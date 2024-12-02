import os, sys
from pathlib import Path
import pandas as pd
import scanpy as sc
import anndata as ad
import decoupler as dc
import warnings

warnings.filterwarnings("ignore")
_, adata_path, ref_path = sys.argv
adata_path = Path(adata_path)
adata = ad.read(adata_path)

# ------------------------------------------
# leiden clustering
sc.tl.leiden(adata)

# ------------------------------------------
# get marker gene reference
markers = pd.read_csv(ref_path)
# Filter by human, canonical_markers, and PBMC 
markers = markers[markers['human'] & markers['canonical_marker'] & (markers['human_sensitivity'] > 0.5) & (markers["organ"].isin({"Blood", "Immune system"}))]
# Remove duplicated entries
markers = markers[~markers.duplicated(['cell_type', 'genesymbol'])]

# ------------------------------------------
# run over representation analysis
adata.var.set_index("gene", inplace=True)
dc.run_ora(
    mat=adata,
    net=markers,
    source='cell_type',
    target='genesymbol',
    min_n=3,
    verbose=True,
    use_raw=False
)
acts = dc.get_acts(adata, obsm_key='ora_estimate')

# ------------------------------------------
# annotate leiden clusters with consensus cell types
df = dc.rank_sources_groups(acts, groupby='leiden', reference='rest', method='t-test_overestim_var')
annotation_dict = df.groupby('group').head(1).set_index('group')['names'].to_dict()
adata.obs['cell_type'] = [annotation_dict[clust] for clust in adata.obs['leiden']]

# ------------------------------------------
# Visualize
ax = sc.pl.umap(adata, color='cell_type', s=5, title="Cell Type", show=False)
ax.figure.set_figwidth(6)
ax.figure.set_figheight(6)

ax.figure.savefig('umap_celltypes.png', dpi=300, bbox_inches='tight')
out_path = f"adata.{Path(__file__).stem}.h5ad"
adata.write(out_path)