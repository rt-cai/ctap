import os, sys
from pathlib import Path
from scipy.io import mmread
import pandas as pd
import numpy as np
import scanpy as sc
import anndata as ad
import warnings

warnings.filterwarnings("ignore")
_, data_dir = sys.argv
data_dir = Path(data_dir)

# ------------------------------------------
# load tables
todo = ["HCMV1", "HCMV2", "mock1", "mock2"]
sample_bars = {}
sample_genes = {}
for sample_name in todo:
    df_gene = pd.read_csv(data_dir/f"{sample_name}-features.tsv", sep="\t", header=None)
    df_gene = df_gene.iloc[:, 0:2]
    df_gene.columns = ["locus", "gene"]
    sample_genes[sample_name] = df_gene

    df_bar = pd.read_csv(data_dir/f"{sample_name}-barcodes.tsv", sep="\t", header=None)
    df_bar = df_bar.iloc[:, 0:1]
    df_bar.columns = ["barcode"]
    sample_bars[sample_name] = df_bar

# ------------------------------------------
# merge using genes as index
def make_index_map(df):
    x2i = pd.DataFrame(df)
    x2i.set_index("locus", inplace=True)
    x2i["ival"] = np.arange(len(df))
    return x2i
_dfs = list(sample_genes.values())
df_allg = pd.DataFrame(_dfs[0])
for _df in _dfs[1:]:
    df_allg = pd.merge(df_allg, _df, how="outer", on="locus")
    df_allg["gene_x"].fillna(df_allg["gene_y"], inplace=True)
    df_allg.columns = [c.replace("_x", "") for c in df_allg.columns]
    df_allg = df_allg[["locus", "gene"]]

# ------------------------------------------
# load count matrix
mats = {}
for sample_name in todo:
    sample = mmread(data_dir/f"{sample_name}-matrix.mtx").T
    genes = sample_genes[sample_name]
    sample_len = sample.shape[0]

    g2i = make_index_map(genes)
    all_genes_f = df_allg[df_allg["locus"].isin(genes["locus"])].locus
    new_gene_order = g2i.loc[all_genes_f].ival.to_numpy(dtype=np.int32)

    sample = sample.tocsr()[:, new_gene_order]
    sample.resize((sample_len, df_allg.shape[0]))
    df_obs = sample_bars[sample_name]
    df_obs["barcode"] = f"{sample_name}-"+df_obs["barcode"]
    mats[sample_name] = sc.AnnData(sample, obs=df_obs)

# ------------------------------------------
# create adata
adata = ad.concat(mats, label="sample")
adata.var = df_allg
adata.obs.columns = ["barcode", "sample"]
rename_dict = {
    "HCMV": "Infected",
    "mock": "Control",
}
rename = lambda x: rename_dict.get(x, x)
adata.obs["sample_type"] = adata.obs["sample"].str[:-1].apply(rename)

# ------------------------------------------
# save
out_path = "adata.initial.h5ad"
adata.write(out_path)
