#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --job-name=bartab_sc_yogesh
#SBATCH --time=00-1:00:00
#SBATCH --mem=24GB
#SBATCH --mail-user='henrietta.holze@petermac.org'
#SBATCH --mail-type=ALL
#SBATCH --output='logs/%x.%j.out'
#SBATCH --error='logs/%x.%j.err'
#SBATCH --partition=prod_short

#### Load modules ####
# Clean environment
module purge
# Load modules
module load singularity/3.7.3
module load nextflow/23.04.1

export NXF_SINGULARITY_LIBRARYDIR="/dawson_genomics/Projects/bartools_bartab_paper/singularity/"    # your singularity storage dir
# export NXF_SINGULARITY_CACHEDIR=MY_SINGULARITY_CACHE       # your singularity cache dir

nextflow run /researchers/henrietta.holze/splintr_tools/BARtab/BARtab.nf \
  -profile singularity \
  -params-file bartab_fm02_sc_amplicon_50bp_no_umi_yogesh_params.yaml \
  -w "/dawson_genomics/Projects/bartools_bartab_paper/results/sc_yogesh_comparison/work/" \
  -resume

