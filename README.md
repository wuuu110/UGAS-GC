## Paper
The source code of "Towards Efficient Few-shot Graph Neural Architecture Search via Partitioning Gradient Contribution".

## Setup
```bash
git clone https://github.com/rampasek/GraphGPS.git
conda create -n UGAS python=3.9
conda activate UGAS
conda install pytorch=1.10 torchvision torchaudio -c pytorch -c nvidia
conda install pyg=2.0.4 -c pyg -c conda-forge
conda install openbabel fsspec rdkit -c conda-forge
pip install torchmetrics
pip install performer-pytorch
pip install ogb
pip install tensorboardX
pip install wandb
conda clean --all
```

## Acknowledgment
This repository is forked from [GraphGPS](https://github.com/rampasek/GraphGPS)