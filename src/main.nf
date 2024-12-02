process extract_hcmv {
    output:
    path "processed_data_files"

    """
    pigz -dc ${params.data}/GSE267869_Processed_data_files.tar.gz | tar -xf -
    mv "Processed data files" processed_data_files
    cd processed_data_files
    ls | xargs -I % pigz -d %
    """
}

process load_initial_adata {
    input:
    path raw

    output:
    path "adata.initial.h5ad"

    """
    python ${params.steps}/${task.process}.py ${raw}
    """
}

process qc {
    publishDir "$params.output", mode: 'copy', pattern: '*.png'

    input:
    path adata

    output:
    path "counts_distribution.png"

    """
    python ${params.steps}/${task.process}.py ${adata}
    """
}

process log_normalize_counts {
    input:
    path adata

    output:
    path "adata.${task.process}.h5ad"

    """
    python ${params.steps}/${task.process}.py ${adata}
    """
}

process umap_embed {
    label 'compute_intensive'
    publishDir "$params.output", mode: 'copy', pattern: '*.png'

    input:
    path adata

    output:
    path "adata.${task.process}.h5ad"
    path "umap_infection_vs_control.png"

    """
    python ${params.steps}/${task.process}.py ${adata}
    """
}

process predict_cell_types {
    publishDir "$params.output", mode: 'copy', pattern: '*.png'
    
    input:
    path adata

    output:
    path "adata.${task.process}.h5ad"
    path "umap_celltypes.png"

    """
    python ${params.steps}/${task.process}.py ${adata} ${params.data}/PanglaoDB.csv
    """
}

process compare_cell_populations {
    publishDir "$params.output", mode: 'copy', pattern: '*.png'
    
    input:
    path adata

    output:
    path "compare_cell_populations.png"

    """
    python ${params.steps}/${task.process}.py ${adata}
    """
}

process publish_adata {
    publishDir "$params.output", mode: 'copy', pattern: '*.h5ad'
    
    input:
    path adata
    val name

    output:
    path "${name}.h5ad"

    """
    mv ${adata} ${name}.h5ad
    """
}

workflow {
    raw = extract_hcmv()
    adata = load_initial_adata(raw)
    fig_qc = qc(adata)
    adata = log_normalize_counts(adata)
    (adata, fig_umap) = umap_embed(adata)
    (adata, fig_celltypes) = predict_cell_types(adata)
    fig_comp = compare_cell_populations(adata)
    publish_adata(adata, "adata.final")
}
