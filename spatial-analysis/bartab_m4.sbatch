#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=24
#SBATCH --job-name=bartab_m4
#SBATCH --time=00-10:00:00
#SBATCH --mem=62GB
#SBATCH --mail-user='henrietta.holze@petermac.org'
#SBATCH --mail-type=ALL
#SBATCH --output='logs/%x.%j.out'
#SBATCH --error='logs/%x.%j.err'
#SBATCH --partition=prod_med

#### Load modules ####
# Clean environment
module purge
# Load modules
module load singularity/3.7.3
module load nextflow/23.04.1

# running bartab with commit 898f74758ce5ea9558beadfdba66c9043e1e3a92 on branch saw_input_fastq and conda env bartab-1.2

export NXF_SINGULARITY_LIBRARYDIR="/scratch/users/hholze/BARtab/singularity/"    # your singularity storage dir
# export NXF_SINGULARITY_CACHEDIR=MY_SINGULARITY_CACHE       # your singularity cache dir

nextflow run /researchers/henrietta.holze/splintr_tools/BARtab/BARtab.nf \
  -profile singularity \
  -params-file /dawson_genomics/Projects/BGI_spatial/scripts/bartab_params_m4.yml \
  -w "/scratch/users/hholze/BGI_spatial/work/" \
  -resume
