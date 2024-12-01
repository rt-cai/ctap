library(Seurat)
library(BPCells)
library(dplyr)
library(ggplot2)
library(ggrepel)
library(patchwork)
# set this option when analyzing large datasets
options(future.globals.maxSize = 3e+09)

parse.mat <- open_matrix_dir(dir = "/brahms/hartmana/vignette_data/bpcells/parse_1m_pbmc")
# need to move
metadata <- readRDS("/brahms/haoy/vignette_data/ParseBio_PBMC_meta.rds")
object <- CreateSeuratObject(counts = parse.mat, meta.data = metadata)

object <- NormalizeData(object)
# split assay into 24 layers
object[["RNA"]] <- split(object[["RNA"]], f = object$sample)
object <- FindVariableFeatures(object, verbose = FALSE)

object <- SketchData(object = object, ncells = 5000, method = "LeverageScore", sketched.assay = "sketch")
object
