#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=20
#SBATCH --job-name=pycashier_yogesh
#SBATCH --time=00-24:00:00
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

# activate conda environment
source activate cashierenv

# pycashier needs unzipped fastq files
# cp bartab_input/* /scratch/users/hholze/BARtab/pycashier_yogesh_fastq/
# gunzip /scratch/users/hholze/BARtab/pycashier_yogesh_fastq/*

{ time ( pycashier extract -c pycashier_yogesh_all_samples.toml -t 20 -y -v ;
       pycashier combine -c pycashier_yogesh_all_samples.toml -v ) } 2> pycashier_yogesh_runtime.txt

