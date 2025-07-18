#!/bin/sh
cd /MLRU++
DATASET_PATH=../DATASET/DATASET_Acdc

export PYTHONPATH=./
export RESULTS_FOLDER=./acdc_INCEPMOA
export unetr_pp_preprocessed=../DATASET/DATASET_Acdc/unetr_pp_raw/unetr_pp_raw_data/Task01_ACDC
#"$DATASET_PATH"/unetr_pp_raw/unetr_pp_raw_data/Task01_ACDC
export unetr_pp_raw_data_base=../DATASET/DATASET_Acdc/unetr_pp_raw

python mlru_pp/run/run_training.py 3d_fullres mlru_pp_trainer_acdc 1 0
