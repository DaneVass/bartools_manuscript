#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=20
#SBATCH --job-name=timemachine_umi_yogesh
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

# activate conda env with python 2.7
source activate timemachine

timemachine_path="/dawson_genomics/Projects/bartools_bartab_paper/scripts/yogesh_comparison/timemachine"

# array of all sample folders
sample_folders=($(ls -lah /dawson_genomics/Projects/bartools_bartab_paper/yogesh_data/gDNA_Barcodes/BC01_BC02_BC03_FM02_FM03/*/ | grep '^/' | sed 's/://g'))

# will save results in working directory
cd /dawson_genomics/Projects/bartools_bartab_paper/results/yogesh_comparison/timemachine/

# echo "\nClustering with UMI counts\n"
# also run UMI quantification to see if we can reproduce results from paper but don't let it count towards timing
for sample_folder in "${sample_folders[@]}"; do
  samplename=$(echo $sample_folder | sed 's/.*\/\(.*\)\//\1/g')
  # date
  # echo "## barcode clustering UMIs $samplename"
  python ${timemachine_path}/stepTwoRunStarcode/submitStarcode.py $samplename --countType UMIs
done

