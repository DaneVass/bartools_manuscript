#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=20
#SBATCH --job-name=bartab_yogesh
#SBATCH --time=00-1:00:00
#SBATCH --mem=32GB
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

export NXF_SINGULARITY_LIBRARYDIR="/scratch/users/hholze/BARtab/singularity/"    # your singularity storage dir
# export NXF_SINGULARITY_CACHEDIR=MY_SINGULARITY_CACHE       # your singularity cache dir

{ time ( nextflow run /researchers/henrietta.holze/splintr_tools/BARtab/BARtab.nf \
  -profile singularity \
  -params-file /dawson_genomics/Projects/bartools_bartab_paper/scripts/yogesh_comparison/bartab_yogesh_all_samples_variable_length_params.yaml \
  -w "/dawson_genomics/Projects/bartools_bartab_paper/results/yogesh_comparison/work/" ) } 2> bartab_yogesh_runtime.txt

