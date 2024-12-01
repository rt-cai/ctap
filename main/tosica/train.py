import os, sys
from pathlib import Path
sys.path = list(set(sys.path+[str(p) for p in [
    "/app",
]]))
TEMP = Path("./tmp/")
TEMP.mkdir(exist_ok=True)
for k, v in [
    ("NUMBA_CACHE_DIR", TEMP/"numba_cache"),
    ("MPLCONFIGDIR", TEMP/"matplotlib"),
    ("XDG_CACHE_HOME", TEMP/"xdg_home"),
]:
    os.environ[k] = str(v)

import TOSICA
import scanpy as sc
import warnings
import torch

print(f"gpu:{torch.cuda.is_available()}, device:{torch.cuda.get_device_name(torch.cuda.current_device())}")
warnings.filterwarnings("ignore")

ref_adata = sc.read('/data/PHCA.h5ad')
print(ref_adata)

# sc.pp.subsample(ref_adata, n_obs=int(ref_adata.shape[0]*0.001), copy=False)
# print(ref_adata)

os.chdir('./cache')
# batch size can not be 1

print(f"""
=============================================
train
=============================================
""")
TOSICA.train(
    ref_adata, gmt_path='human_gobp',project ='train_test', label_name='Manually_curated_celltype',
    epochs=8, batch_size=2, lr=0.001, embed_dim=128,
)
