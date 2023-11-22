#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --job-name=download_saw_singularity
#SBATCH --time=00-08:00:00
#SBATCH --mem=32GB
#SBATCH --mail-user='henrietta.holze@petermac.org'
#SBATCH --mail-type=ALL
#SBATCH --output='%j.out'
#SBATCH --error='%j.err'
#SBATCH --partition=prod

#### Load modules ####
# Clean environment
module purge
# load modules
module load singularity/3.7.3

# following installation instructions in https://github.com/BGIResearch/SAW
# downloading singularity image to run SAW pipeline on BGI samples
# download to project directory so it can be used by multiple people

export SINGULARITY_CACHEDIR=/dawson_genomics/Projects/BGI_spatial/.singularity/cache/
singularity build SAW_06.1.0.sif docker://stomics/saw:06.1.0
