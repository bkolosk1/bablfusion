#!/bin/bash
#SBATCH --output=outs/tpot_output_%j.txt
#SBATCH --error=err/tpot_error_%j.txt
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

data_folder=$1
dataset_name=$2
mode=$3
proj=$4

job_name="tpot_${dataset_name}_mode${mode}_proj${proj}"
#SBATCH --job-name=$job_name

SINGULARITY_IMAGE=experiment_container.sif

singularity exec --bind ./export:/export $SINGULARITY_IMAGE python3 tpot_code.py $data_folder $dataset_name $mode $proj