# MLRU++: Multiscale Lightweight Residual UNETR++ with Attention for Efficient 3D MedicalImage Segmentation




## Results


## Qualitative Comparison

### Synapse Dataset
Qualitative comparison on multi-organ segmentation 3D task.
![Synapse Qual Results](media/MLUN++.png)




<hr />

## Installation
The code is tested with PyTorch 1.11.0 and CUDA 11.3. After cloning the repository, follow the below steps for installation,

1. Create and activate conda environment
```shell
conda create --name unetr_pp python=3.8
conda activate unetr_pp
```
2. Install PyTorch and torchvision
```shell
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
```
3. Install other dependencies
```shell
pip install -r requirements.txt
```
<hr />


## Dataset
We follow the same dataset preprocessing as in [nnFormer](https://github.com/282857341/nnFormer). We conducted extensive experiments on five benchmarks: Synapse, BTCV, ACDC, BRaTs, and Decathlon-Lung. 

The dataset folders for Synapse should be organized as follows: 

```
./DATASET_Synapse/
  ├── unetr_pp_raw/
      ├── unetr_pp_raw_data/
           ├── Task02_Synapse/
              ├── imagesTr/
              ├── imagesTs/
              ├── labelsTr/
              ├── labelsTs/
              ├── dataset.json
           ├── Task002_Synapse
       ├── unetr_pp_cropped_data/
           ├── Task002_Synapse
 ```
 
 The dataset folders for ACDC should be organized as follows: 

```
./DATASET_Acdc/
  ├── unetr_pp_raw/
      ├── unetr_pp_raw_data/
           ├── Task01_ACDC/
              ├── imagesTr/
              ├── imagesTs/
              ├── labelsTr/
              ├── labelsTs/
              ├── dataset.json
           ├── Task001_ACDC
       ├── unetr_pp_cropped_data/
           ├── Task001_ACDC
 ```
 
  The dataset folders for Decathlon-Lung should be organized as follows: 

```
./DATASET_Lungs/
  ├── unetr_pp_raw/
      ├── unetr_pp_raw_data/
           ├── Task06_Lung/
              ├── imagesTr/
              ├── imagesTs/
              ├── labelsTr/
              ├── labelsTs/
              ├── dataset.json
           ├── Task006_Lung
       ├── unetr_pp_cropped_data/
           ├── Task006_Lung
 ```

 
Please refer to [Setting up the datasets](https://github.com/282857341/nnFormer) on nnFormer repository for more details.

## Training
The following scripts can be used for training our UNETR++ model on the datasets:
```shell
bash training_scripts/run_training_synapse.sh
bash training_scripts/run_training_acdc.sh
bash training_scripts/run_training_lung.sh
bash training_scripts/run_training_tumor.sh
```

<hr />

## Evaluation

To reproduce the results of MLRU++;
Soon will share the weights using google drive link



<hr />

## Acknowledgement
This repository is built based on [nnFormer](https://github.com/282857341/nnFormer) repository.


