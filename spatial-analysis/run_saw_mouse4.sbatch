#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=20
#SBATCH --job-name=saw_m4
#SBATCH --time=02-00:00:00
#SBATCH --mem=160GB
#SBATCH --mail-user='henrietta.holze@petermac.org'
#SBATCH --mail-type=ALL
#SBATCH --output='R-%x.%j.out'
#SBATCH --error='R-%x.%j.err'
#SBATCH --partition=prod

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
reference_dir="${project_dir}/raw_data/reference/m38"

# for saw_v6.0.0
sh ${project_dir}/scripts/SAW_pipeline/SAW/Scripts/stereoPipeline_v6.1.sh -genomeSize 3 \
  -splitCount 1 \
  -maskFile ${project_dir}/raw_data/05_P21Z19700N0017-T3_PeterMacMouse/Mouse_4_spleen/01.Filter/SS200000412TL_C2.barcodeToPos.h5 \
  -fq1 ${project_dir}/raw_data/04_P21Z19700N0017-T3_PeterMacMouse/Mouse_4_spleen/01.Filter/DP8400029990TL_L01_read_1.fq.gz \
  -fq2 ${project_dir}/raw_data/04_P21Z19700N0017-T3_PeterMacMouse/Mouse_4_spleen/01.Filter/DP8400029990TL_L01_read_2.fq.gz \
  -imageRecordFile ${project_dir}/preprocessed_data/imageStudio_output/QC/SS200000412TL_C2_SC_20230602_124957_2.0.1.ipr \
  -imageCompressedFile ${project_dir}/preprocessed_data/imageStudio_output/QC/SS200000412TL_C2_SC_20230602_124957_2.0.1.tar.gz \
  -refIndex ${reference_dir}/index \
  -genomeFile /data/reference/indexes/mouse/ensembl_GRCm38.78/fasta/Mus_musculus.GRCm38.dna.toplevel.fa \
  -speciesName mouse \
  -tissueType spleen \
  -annotationFile ${reference_dir}/Mus_musculus.GRCm38.78.filtered.gtf \
  -outDir ${project_dir}/preprocessed_data/saw_output/mouse4_spleen/image_no_cellbin/ \
  -doCellBin N \
  -threads 20 \
  -sif ${project_dir}/SAW_06.1.0.sif
