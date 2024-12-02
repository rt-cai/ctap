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

process load_adata_from_tables {
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
    path "qc_metrics.png"

    """
    python ${params.steps}/${task.process}.py ${adata}
    """
}

process log_norm {
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
    path "umap_infection.png"

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

process compare_infection {
    publishDir "$params.output", mode: 'copy', pattern: '*.png'
    
    input:
    path adata

    output:
    path "compare_infection.png"

    """
    export PYTHONPATH=${projectDir}:PYTHONPATH
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
    adata = load_adata_from_tables(raw)
    fig_qc = qc(adata)
    adata = log_norm(adata)
    (adata, fig_umap) = umap_embed(adata)
    (adata, fig_celltypes) = predict_cell_types(adata)
    fig_comp = compare_infection(adata)
    publish_adata(adata, "adata.final")
}


// workflow {
//     data_hcmv = path('https://ftp.ncbi.nlm.nih.gov/geo/series/GSE267nnn/GSE267869/suppl/GSE267869%5FProcessed%5Fdata%5Ffiles.tar.gz')
//     def steps_path = System.getenv('CTAP_LIB')
//     println(steps_path)
// }

// process peek {
//     input:
//     val x

//     output:
//     stdout
//     """
//     echo $params.x
//     echo $x
//     """
// }

// process a1 {
//     publishDir "output", mode: 'copy'
    
//     output:
//     path "out.x"

//     """
//     mkdir -p output
//     echo "$params" > out.x
//     """
// }

// process a2 {
//     input:
//     path x

//     output:
//     stdout

//     """
//     cat $x
//     """
// }

// workflow {
//     a1 | a2 | view
// }
