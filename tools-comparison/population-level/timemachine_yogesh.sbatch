#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=20
#SBATCH --job-name=timemachine_yogesh
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

#### get stagger information for each sample before hand
# for filename in bartab_input/*001.fastq.gz; do
#   stagger=$(zgrep -o -E ".*TCGACTAAACGCGCTACTTGAT" $filename | head -n1 | awk '{print length}')
#   echo "$filename $(($stagger - 6 - 22))" | sed 's/bartab_input\///g;s/_L.*gz//g'
# done | uniq > yogesh_stagger.txt

# !! need to manually edit BC03 sample names


# activate conda env, created with: 
# conda create -n timemachine -c conda-forge -c bioconda python=2.7 biopython numpy regex pandas starcode
source activate timemachine

# github repo with timemachine code, see fork https://github.com/HenriettaHolze/timemachine
timemachine_path="/dawson_genomics/Projects/bartools_bartab_paper/scripts/yogesh_comparison/timemachine"

# array of all sample folders
sample_folders=($(ls -lah /dawson_genomics/Projects/bartools_bartab_paper/yogesh_data/gDNA_Barcodes/BC01_BC02_BC03_FM02_FM03/*/ | grep '^/' | sed 's/://g'))

# will save results in working directory
cd /dawson_genomics/Projects/bartools_bartab_paper/results/yogesh_comparison/timemachine/

{ time for sample_folder in "${sample_folders[@]}"; do
  samplename=$(echo $sample_folder | sed 's/.*\/\(.*\)\//\1/g')
  stagger=$(echo $samplename | sed 's/_/-/g' | grep -f - /dawson_genomics/Projects/bartools_bartab_paper/scripts/yogesh_comparison/yogesh_stagger.txt | cut -d" " -f2)
  # date
  # echo "## barcode extraction $samplename"
  # write separate files for read and UMI counts 
  python ${timemachine_path}/stepOneExtractBarcodes/timeMachineParseFastQ150SEreads_v4.py $sample_folder -o $samplename -s $stagger --includeReads
  # echo "## barcode clustering reads $samplename"
  python ${timemachine_path}/stepTwoRunStarcode/submitStarcode.py $samplename
done } 2> timemashine_yogesh_runtime.txt

