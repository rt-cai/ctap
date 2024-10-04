# Introduction

### *Background & Rationale*
Single cell RNA (scRNA) sequencing enables the viewing of gene expression patterns on a cell by cell basis, which communicates the function and intent of individual cells from within a possibly heterogeneous population. Observing the space of possible expression patterns will reveal prevalent distributions, otherwise known as cell types. It is important to model cell types not as distinct categories, but rather as relatively dense regions within an otherwise continuous landscape. Time-aware pathfinding between cell types can be used to infer the developmental trajectories of cells as they reach maturity from stem cells to senescence. A continuous model also provides the resolution to map expression responses from environmental stimuli, perturbations, that are distinct from differences in cell type. This ability to identify cell types and adjacent gene expression patterns is applicable to a wide variety of situations, including forensics [(1)](#bibliography), pathogen detection [(2)](#bibliography), and monitoring disease progress [(3)](#bibliography). Perhaps the simplest method for assigning cell types is through the use of canonical cell surface markers during flow cytometry[(4)](#bibliography), but more sophisticated tools have since been developed to integrate more genes into the classification process. Seurat [(5)](#bibliography) and Monocle [(6)](#bibliography) are examples of popular toolkits for this purpose, integrating many statistical, heuristic, and machine learning modules. With the widespread success of deep learning approaches centered around the breakthrough technology of the attention mechanism and transformers [(7)](#bibliography), modern approaches like TOSICA [(8)](#bibliography) and scGPT [(9)](#bibliography) have pushed the performance in cell type mapping even further. 

### *Aims*
The purpose of this pipeline is to demonstrate and automate a cell type annotation protocol using state of the art methods.

### *Dependencies*

### *Workflow*

# Usage

### *Installation*

### *Requried Inputs*

### *Execution*

### *Expected Outputs*

### *CLI Reference*

# Bibliography

1.	Sijen, T. Molecular approaches for forensic cell type identification: On mRNA, miRNA, DNA methylation and microbial markers. Forensic Sci. Int. Genet. 18, 21–32 (2015). [https://doi.org/10.1016/j.fsigen.2014.11.015](https://doi.org/10.1016/j.fsigen.2014.11.015)
2.	Huang, W., Wang, D. & Yao, Y.-F. Understanding the pathogenesis of infectious diseases by single-cell RNA sequencing. Microb. Cell 8, 208–222. [https://doi.org/10.15698%2Fmic2021.09.759](https://doi.org/10.15698%2Fmic2021.09.759)
3.	Ding, S., Chen, X. & Shen, K. Single-cell RNA sequencing in breast cancer: Understanding tumor heterogeneity and paving roads to individualized therapy. Cancer Commun. 40, 329–344 (2020). [https://doi.org/10.1002/cac2.12078](https://doi.org/10.1002/cac2.12078)
4.	McKinnon, K. M. Flow Cytometry: An Overview. Curr. Protoc. Immunol. 120, 5.1.1-5.1.11 (2018). [https://doi.org/10.1002%2Fcpim.40](https://doi.org/10.1002%2Fcpim.40)
5.	Butler, A., Hoffman, P., Smibert, P., Papalexi, E. & Satija, R. Integrating single-cell transcriptomic data across different conditions, technologies, and species. Nat. Biotechnol. 36, 411–420 (2018). [https://doi.org/10.1038/nbt.4096](https://doi.org/10.1038/nbt.4096)
6.	Trapnell, C. et al. The dynamics and regulators of cell fate decisions are revealed by pseudotemporal ordering of single cells. Nat. Biotechnol. 32, 381–386 (2014). [https://doi.org/10.1038/nbt.2859](https://doi.org/10.1038/nbt.2859)
7.	Vaswani, A. et al. Attention Is All You Need. Preprint at [https://doi.org/10.48550/arXiv.1706.03762](https://doi.org/10.48550/arXiv.1706.03762) (2023).
8.	Chen, J. et al. Transformer for one stop interpretable cell type annotation. Nat. Commun. 14, 223 (2023). [https://doi.org/10.1038/s41467-023-35923-4](https://doi.org/10.1038/s41467-023-35923-4)
9.	Cui, H. et al. scGPT: Towards Building a Foundation Model for Single-Cell Multi-omics Using Generative AI. 2023.04.30.538439 Preprint at [https://doi.org/10.1101/2023.04.30.538439](https://doi.org/10.1101/2023.04.30.538439) (2023).
