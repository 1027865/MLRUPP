#!/bin/sh
cd /home/rizk_lab/shared/USDN/MED/M3D_CAP/Brats2017/MLRU++
DATASET_PATH=../DATASET/DATASET_Tumor

export PYTHONPATH=./
export RESULTS_FOLDER=./output_tumor2
export unetr_pp_preprocessed=../DATASET/DATASET_Tumor/DATASET_Tumor/unetr_pp_raw/unetr_pp_raw_data/Task03_tumor
export unetr_pp_raw_data_base=../DATASET/DATASET_Tumor/unetr_pp_raw

python mlru_pp/run/run_training.py 3d_fullres unetr_pp_trainer_tumor 3 0
