#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=10
#SBATCH --job-name=lasso
#SBATCH --time=00-02:00:00
#SBATCH --mem=16GB
#SBATCH --mail-user='henrietta.holze@petermac.org'
#SBATCH --mail-type=ALL
#SBATCH --output='R-%x.%j.out'
#SBATCH --error='R-%x.%j.err'
#SBATCH --partition=prod_short

#### Load modules ####
# Clean environment
module purge
# load modules
module load singularity/3.7.3

export SINGULARITY_CACHEDIR=/dawson_genomics/Projects/BGI_spatial/.singularity/cache/

ulimit -n 10240
ulimit -v 33170449147
NUMBA_CACHE_DIR=/dawson_genomics/Projects/BGI_spatial/

project_dir="/dawson_genomics/Projects/BGI_spatial"

export SINGULARITY_BIND="${project_dir}/preprocessed_data/saw_output/,${project_dir}/preprocessed_data/StereoMapOutput/"

# lasso in StereoMap was with bin10

singularity exec ${project_dir}/SAW_06.1.0.sif lasso \
    -i ${project_dir}/preprocessed_data/saw_output/mouse4_spleen/image_no_cellbin/04.tissuecut/SS200000412TL_C2.gef \
    -m ${project_dir}/preprocessed_data/StereoMapOutput/20230801184452.whole_tissue_mouse4_lasso.geojson \
    -o ${project_dir}/preprocessed_data/saw_output/mouse4_spleen/image_no_cellbin/lasso/ \
    -s 10 \
    -n SS200000412TL_C2

