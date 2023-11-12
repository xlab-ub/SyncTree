# SyncTree

This repo implements the experiments in 2023 NeurIPS paper "SyncTREE: Fast Timing Analysis for Integrated Circuit Design through a Physics-informed Tree-based Graph Neural Network".

## Installation

1.Install [Anaconda](https://www.anaconda.com/download).

Create conda env

```bash
conda env create -f environment.yml
```

2.Install [pygraphviz](https://pygraphviz.github.io/documentation/stable/install.html). 

## Download Dataset

1.Download Raw RC-tree json files.

Google Drive: https://drive.google.com/file/d/1TEux9yTVc2--zC-__4qbd2MJQJdSGzof/view?usp=drive_link

OR

```bash
pip install gdown
gdown 1TEux9yTVc2--zC-__4qbd2MJQJdSGzof
tar -zxvf rawdata.tar.gz
```

2.Download processed graph files.

Google Drive: https://drive.google.com/file/d/1S4g8cYjFqOTxjGYRjzYCJIor5_4Bvlss/view?usp=sharing

OR 

```bash
pip install gdown
gdown 1S4g8cYjFqOTxjGYRjzYCJIor5_4Bvlss
tar -zxvf traindata.tar.gz
```
## Replace gat_conv.py 
Replace anaconda_dir/envs/spice/lib/python3.9/site-packages/torch_geometric/nn/conv/gat_conv.py with gat_conv.py.

for example
```bash
cd GNN2SPICE
mv gat_conv.py /miniconda3/envs/spice/lib/python3.9/site-packages/torch_geometric/nn/conv/
```

## Run Experiments
Adjust hyper-parameters in parameters.yaml and specify in main.py
```bash
python main.py
```
