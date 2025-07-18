#!/bin/sh
cd /home/rizk_lab/shared/USDN/MED/M3D_CAP/Brats2017/MLRU


DATASET_PATH=../DATASET/DATASET_Lungs

export PYTHONPATH=./
export RESULTS_FOLDER=./output_lung_BEST
export unetr_pp_preprocessed=../DATASET/DATASET_Lungs/DATASET_Lungs/unetr_pp_raw/unetr_pp_raw_data/Task06_Lung
export unetr_pp_raw_data_base=./DATASET/DATASET_Lungs/unetr_pp_raw

python mlru_pp/run/run_training.py 3d_fullres mlru_pp_trainer_lung 6 0
