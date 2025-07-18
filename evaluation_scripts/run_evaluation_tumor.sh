#!/bin/sh

DATASET_PATH=../DATASET_Tumor

export PYTHONPATH=.././
export RESULTS_FOLDER=../mlru_pp/evaluation/mlru_pp_tumor_checkpoint
export unetr_pp_preprocessed="$DATASET_PATH"/unetr_pp_raw/unetr_pp_raw_data/Task03_tumor
export unetr_pp_raw_data_base="$DATASET_PATH"/unetr_pp_raw


# Only for Tumor, it is recommended to train unetr_plus_plus first, and then use the provided checkpoint to evaluate. It might raise issues regarding the pickle files if you evaluated without training

python ../mlru_pp/inference/predict_simple.py -i 
python ../mlru_pp/inference_tumor.py 0

