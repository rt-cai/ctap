# Introduction

### *Background & Rationale*
Single cell RNA (scRNA) sequencing enables the viewing of gene expression patterns on a cell by cell basis, which communicates the function and intent of individual cells from within a possibly heterogeneous population. Observing the space of possible expression patterns will reveal prevalent distributions, otherwise known as cell types. It is important to model cell types not as distinct categories, but rather as relatively dense regions within an otherwise continuous landscape. Time-aware pathfinding between cell types can be used to infer the developmental trajectories of cells as they reach maturity from stem cells to senescence. A continuous model also provides the resolution to map expression responses from environmental stimuli, perturbations, that are distinct from differences in cell type. This ability to identify cell types and adjacent gene expression patterns is applicable to a wide variety of situations, including forensics [(1)](#bibliography), pathogen detection [(2)](#bibliography), and monitoring disease progress [(3)](#bibliography). Perhaps the simplest method for assigning cell types is through the use of canonical cell surface markers during flow cytometry[(4)](#bibliography), but more sophisticated tools have since been developed to integrate more genes into the classification process. Seurat [(5)](#bibliography) and Monocle [(6)](#bibliography) are examples of popular toolkits for this purpose, integrating many statistical, heuristic, and machine learning modules. With the widespread success of deep learning approaches centered around the breakthrough technology of the attention mechanism and transformers [(7)](#bibliography), modern approaches like TOSICA [(8)](#bibliography) and scGPT [(9)](#bibliography) have pushed the performance in cell type mapping even further. 

Transcriptomics count data provided by Wang and colleagues as part of the publication [(10)](#bibliography)

### *Aims*
The purpose of this pipeline is to demonstrate and automate a cell type annotation protocol using state of the art methods.

### *Dependencies*

### *Workflow*

# Usage

### *Installation*

### *Requried Inputs*

### *Execution*

### *Expected Outputs*

### *CLI Reference

# Input


```
GSE267869/
├─ {sample}-barcodes.tsv
├─ {sample}-features.tsv
├─ {sample}-matrix.mtx
```
where `sample`  

# Output

```
ctap_results/
├─ adata.final.h5ad
├─ qc_metrics.png
├─ umap_infection.png
├─ umap_celltypes.png
├─ compare_infection.png
```

# Bibliography

1.	Sijen T. Molecular approaches for forensic cell type identification: On mRNA, miRNA, DNA methylation and microbial markers. Forensic Sci Int Genet. 2015;18:21–32. https://doi.org/10.1016/j.fsigen.2014.11.015
2.	Huang W, Wang D, Yao YF. Understanding the pathogenesis of infectious diseases by single-cell RNA sequencing. Microb Cell. 2021;8(9):208–22. https://doi.org/10.15698/mic2021.09.759
3.	Ding S, Chen X, Shen K. Single-cell RNA sequencing in breast cancer: Understanding tumor heterogeneity and paving roads to individualized therapy. Cancer Commun. 2020;40(8):329–44. https://doi.org/10.1002/cac2.12078
4.	McKinnon KM. Flow Cytometry: An Overview. Curr Protoc Immunol. 2018;120(1):5.1.1-5.1.11. https://doi.org/10.1002/cpim.40
5.	Butler A, Hoffman P, Smibert P, Papalexi E, Satija R. Integrating single-cell transcriptomic data across different conditions, technologies, and species. Nat Biotechnol. 2018;36(5):411–20. https://doi.org/10.1038/nbt.4096
6.	Trapnell C, Cacchiarelli D, Grimsby J, Pokharel P, Li S, Morse M, et al. The dynamics and regulators of cell fate decisions are revealed by pseudotemporal ordering of single cells. Nat Biotechnol. 2014;32(4):381–6. https://doi.org/10.1038/nbt.2859
7.	Vaswani A, Shazeer N, Parmar N, Uszkoreit J, Jones L, Gomez AN, et al. Attention is all you need. 2017.  https://doi.org/10.48550/arXiv.1706.03762
8.	Chen J, Xu H, Tao W, Chen Z, Zhao Y, Han JDJ. Transformer for one stop interpretable cell type annotation. Nat Commun. 2023;14(1):223. https://doi.org/10.1038/s41467-023-35923-4
9.	Cui H, Wang C, Maan H, Pang K, Luo F, Duan N, et al. scGPT: toward building a foundation model for single-cell multi-omics using generative AI. Nat Methods. 2024;21(8):1470–80. https://doi.org/10.1038/s41592-024-02201-0
10.	Wang A, Zhu XX, Bie Y, Zhang B, Ji W, Lou J, et al. Single-cell RNA-sequencing reveals a profound immune cell response in human cytomegalovirus-infected humanized mice. Virol Sin. 2024;39(5):782–92. https://doi.org/10.1016/j.virs.2024.08.006