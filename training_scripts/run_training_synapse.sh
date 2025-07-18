#!/bin/sh
cd /MLRU++
DATASET_PATH=../DATASET_Synapse

export PYTHONPATH=./
export RESULTS_FOLDER=./output_Synapse_withour_unetres
export unetr_pp_preprocessed=../DATASET/DATASET_Synapse/unetr_pp_raw/unetr_pp_raw_data/Task02_Synapse
export unetr_pp_raw_data_base="$DATASET_PATH"/unetr_pp_raw

python mlru_pp/run/run_training.py  3d_fullres mlru_pp_trainer_synapse 2 0
