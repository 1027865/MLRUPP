#!/bin/sh
cd /home/rizk_lab/shared/USDN/MED/M3D_CAP/Brats2017/MLRU++
DATASET_PATH=../DATASET/DATASET_BTVC

export PYTHONPATH=./
export RESULTS_FOLDER=./output_BTVC22
export unetr_pp_preprocessed=../DATASET/DATASET_BTVC/unetr_pp_raw/unetr_pp_raw_data/Task02_BTVC
export unetr_pp_raw_data_base=../DATASET/DATASET_BTVC/unetr_pp_raw

python mlru_pp/run/run_training.py 3d_fullres unetr_pp_trainer_synapse 2  0 
