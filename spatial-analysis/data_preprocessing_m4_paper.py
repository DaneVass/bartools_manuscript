#!/usr/bin/env python3

import stereo as st
import scanpy as sc
import utils_stereoseq as us
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import squidpy as sq


def bin_trascripts_barcodes(bin_size, gem_path_m4, gef_path_m4, bc_path_m4, out_path):
    print(f"binning transcripts and barcodes {bin_size}")
    gem = st.io.read_gem(file_path=gem_path_m4, bin_size=bin_size)

    gem.tl.raw_checkpoint()
    adata = st.io.stereo_to_anndata(data=gem, flavor="scanpy")
    adata.uns["attr"] = gem.attr

    bc_data = pd.read_csv(bc_path_m4, sep="\t")
    bc_data = bc_data.rename(columns={"gene": "barcode"})

    # returns table with "barcode", "count_binned", "bin_x", "bin_y", "cell_id", "x_center", "y_center"
    bc_data = us.bin_barcodes(
        bc_data,
        minX=adata.uns["attr"]["minX"],
        minY=adata.uns["attr"]["minY"],
        gef_raw_path=gef_path_m4,
        bin_size=bin_size
    )
    
    # already check if spot is in adata, for QC plot
    bc_data["isin_adata"] = bc_data["cell_id"].isin(adata.obs.index.values)
    
    # save barcode data with all barcodes per bin, and all bins
    bc_data.to_csv(f"{out_path}/mouse4_bin{bin_size}_bc_counts.tsv", sep="\t")
    
    # only keep most abundant barcode per bin
    bc_data = bc_data.sort_values("count_binned", ascending=False).groupby('cell_id').head(1)

    # create column to merge on
    adata.obs["cell_id"] = adata.obs.index.values

    # merge
    adata.obs = adata.obs.reset_index().merge(
        bc_data[["cell_id", "barcode", "count_binned"]], 
        on="cell_id", 
        how="left"
    ).set_index('index')


    sc.pp.calculate_qc_metrics(adata, log1p=False, inplace=True, use_raw=True)

    # save joint data
    print(f"writing output {bin_size}")
    adata.write(filename=f"{out_path}/mouse4_bin{bin_size}_bc.h5ad")

    # export barcode counts on tissue section for bar plot
    adata.obs["barcode"][~adata.obs["barcode"].isna()].value_counts().to_csv(
        f"{out_path}/mouse4_bin{bin_size}_bc_counts_top1.tsv", sep="\t"
    )
    return(adata)


####################################

def filter_cluster(adata, bin_size, out_path, min_counts, max_counts = False):
    print(f"filter cells {bin_size}")
    sc.pp.filter_cells(adata, min_counts=min_counts)
    if max_counts:
        sc.pp.filter_cells(adata, max_counts=max_counts)

    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata)
    adata.raw = adata

    print(f"scale {bin_size}")
    if bin_size <= 10:
        # uses too much memory to zero center
        sc.pp.scale(adata, zero_center=False)
    else:
        sc.pp.scale(adata)

    print(f"pca {bin_size}")
    sc.tl.pca(adata, svd_solver='arpack')

    sc.pp.neighbors(adata)

    print(f"umap {bin_size}")
    sc.tl.umap(adata)
    
    print(f"leiden {bin_size}")
    sc.tl.leiden(adata, resolution=1, key_added="leiden_1")
    sc.tl.leiden(adata, resolution=0.7, key_added="leiden_0.7")
    sc.tl.leiden(adata, resolution=0.5, key_added="leiden_0.5")
    sc.tl.leiden(adata, resolution=0.2, key_added="leiden_0.2")

    print(f"DEG {bin_size}")
    sc.tl.rank_genes_groups(adata, 'leiden_0.7', method='t-test', use_raw=True)

    # save adata
    adata.write(filename=f"{out_path}/mouse4_bin{bin_size}_bc_clustered.h5ad")

    
gem_path_m4 = "/dawson_genomics/Projects/BGI_spatial/preprocessed_data/saw_output/mouse4_spleen/image_no_cellbin/lasso/segmentation/SS200000412TL_C2.lasso.mouse4_whole_tissue.gem.gz"
gef_path_m4 = "/dawson_genomics/Projects/BGI_spatial/preprocessed_data/saw_output/mouse4_spleen/image_no_cellbin/04.tissuecut/SS200000412TL_C2.gef"
bc_path_m4 = "/dawson_genomics/Projects/BGI_spatial/preprocessed_data/splintr_preprocessing/bartab/mouse4_spleen/counts/DP8400029990TL_L01_read_1.unmapped_reads.counts.tsv"

out_path = "/dawson_genomics/Projects/BGI_spatial/plots_paper/input_data_revision/"

# QC thresholds for bin size 20 and 40 are estimated based on histograms of bin size 10 and 50
bin_size = 50
adata = bin_trascripts_barcodes(bin_size, gem_path_m4, gef_path_m4, bc_path_m4, out_path)
filter_cluster(adata, bin_size, out_path, min_counts=600)

bin_size = 40
adata = bin_trascripts_barcodes(bin_size, gem_path_m4, gef_path_m4, bc_path_m4, out_path)
filter_cluster(adata, bin_size, out_path, min_counts=400)

bin_size = 20
adata = bin_trascripts_barcodes(bin_size, gem_path_m4, gef_path_m4, bc_path_m4, out_path)
filter_cluster(adata, bin_size, out_path, min_counts=100)

bin_size = 10
adata = bin_trascripts_barcodes(bin_size, gem_path_m4, gef_path_m4, bc_path_m4, out_path)
filter_cluster(adata, bin_size, out_path, min_counts=30, max_counts = 600)
