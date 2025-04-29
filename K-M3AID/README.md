# K-M3AID

## 1. CLMAModels Packages

### 1) CLMAModel Class

It is a composition of Encoders, Projections,
along with the definition of loss function and optimization algo.


### 2) Encoder Class

It acts as an interface to allow you to load unimodal encoders simply.

### 3) Projection Class

It projects mr1_features and mr2_features into the same vector space,
resulting in the final mr1_embeddings and mr2_embeddings.

### 1) CLMAModel_SP_WP Class
It designed for strong pair and weak pair experiments.

## 2. GraphModels

### 1) NG model 
A general framework of GNN. Input atom feature includes atomic number, chirality tag and hybridization,
while input bond feature includes bond type and bond stereo direction.

Convolution options: `gcn, gin, gat, graphsage, nnconv`. \
Neighborhood aggregation options: `add, mean, max`. \
Jump knowledge (residual connection) options: `last, max, sum`. \
Global pooling options: `add, mean, max, attention`.

## 3. SequenceModels

### 1) CNMRModel 
https://github.com/Qihoo360/CReSS/tree/master

### 2) PeakEncoder
It takes peak position and multiplicity. 

## 4.DatasetModels

### 1) CLMADataset

### 2) GraphDataset

### 3) CNMRDataset 

### 4）PeakDataset 

### 5) PeakDataset_SP_WP
It designed for strong pair and weak pair experiments.

## 5.Datasets

### 1) Training and Validation
It was collected from nmrshiftdb2.

### 2) 1M molecules
It was collected from pubchem

## 6.Utils

### 1) CMRPConfig

It simplifies the way that you modify variables regarding your model.

### 2) BuildDatasetLoader_5fold

It takes dataframe(containing all the files), return train_dataset_loader
and valid_dataset_loader

### 3) AvgMeter

It helps to  keep track of average loss over a period of time and learning_rate,
during the training or evaluation of a model.

### 4) TrainEpoch

It defines how the model get trained.

### 5) ValidEpoch

It defines how the model get validated.

### 6) mr2mr
It computes the dynamic accuracy. 

## 7. Package Installation:

pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.13.0+cu117.html
pip install pytorch_lightning
pip install -U albumentations
pip install timm
pip install rdkit-pypi
pip install pandas
pip install nmrglue

