#!/bin/bash

dataset_paths=("export/embs_aligne" "export/embs_matryoshka_1024" "export/embs_oa_large" "export/embs_oa_small" "export/llm2vec_llama3_1024") 
dataset_names=("books" "dvd" "music" "xgenre" "mldoc" "hatespeech")

modes=(0 1 2)
projections=(0 32)

for i in "${!dataset_paths[@]}"; do
    for j in "${!dataset_names[@]}"; do
        for mode in "${modes[@]}"; do
            for proj in "${projections[@]}"; do
                echo "Submitting path: ${dataset_paths[i]}, name: ${dataset_names[j]}, mode: ${mode}, proj: ${proj}"
                sbatch submit_tpot_job.sh "${dataset_paths[i]}" "${dataset_names[j]}" $mode $proj
            done
        done
    done
done