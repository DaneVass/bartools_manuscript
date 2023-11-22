#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=20
#SBATCH --job-name=build_ref_index
#SBATCH --time=00-10:00:00
#SBATCH --mem=64GB
#SBATCH --mail-user='henrietta.holze@petermac.org'
#SBATCH --mail-type=ALL
#SBATCH --output='R-%x.%j.out'
#SBATCH --error='R-%x.%j.err'
#SBATCH --partition=prod

#### Load modules ####
# Clean environment
module purge
# load modules
module load cellranger/3.1.0
module load singularity/3.7.3

# building index of m10 (mm38) reference genome with SAW pipeline
# building the same way as for analysis of scRNA-seq data, i.e. filtering for protein coding genes

reference_dir="/dawson_genomics/Projects/BGI_spatial/raw_data/reference/m38"

mkdir $reference_dir

ln -s /data/reference/indexes/mouse/ensembl_GRCm38.78/fasta/Mus_musculus.GRCm38.dna.toplevel.fa ${reference_dir}/Mus_musculus.GRCm38.dna.toplevel.fa

cellranger mkgtf \
  /data/reference/indexes/mouse/ensembl_GRCm38.78/bowtie2/Tophat2_transcriptome_data/Mus_musculus.GRCm38.78.gtf \
  ${reference_dir}/Mus_musculus.GRCm38.78.filtered.gtf \
  --attribute=gene_biotype:protein_coding

export SINGULARITY_CACHEDIR=/dawson_genomics/Projects/BGI_spatial/.singularity/cache/

# mount paths to input and output
export SINGULARITY_BIND="${reference_dir},/data/reference/indexes/mouse/ensembl_GRCm38.78/fasta/"

mkdir /dawson_genomics/Projects/BGI_spatial/raw_data/reference/m38/index/

singularity exec /dawson_genomics/Projects/BGI_spatial/SAW_06.1.0.sif mapping --runMode genomeGenerate \
    --genomeDir ${reference_dir}/index \
    --genomeFastaFiles ${reference_dir}/Mus_musculus.GRCm38.dna.toplevel.fa \
    --sjdbGTFfile ${reference_dir}/Mus_musculus.GRCm38.78.filtered.gtf \
    --sjdbOverhang 99 \
    --runThreadN 20
