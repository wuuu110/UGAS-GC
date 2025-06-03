## Introduction
The source code for the paper "[Towards Efficient Few-shot Graph Neural Architecture Search via Partitioning Gradient Contribution](https://arxiv.org/pdf/2506.01231)", accepted by ACM SIGKDD 2025.


# Citation
If you find our paper and code useful, please cite our paper  🚀🚀🚀:
```
@inproceedings{Hercules,
title = {Towards Efficient Few-shot Graph Neural Architecture Search via Partitioning Gradient Contribution},
author = {Song, Wenhao and Wu, Xuan and Yang, Bo and Zhou, You and Xiao, Yubin and Liang, Yanchun and Ge, Hongwei and Lee, Heow Pueh and  Wu, Chunguo},
booktitle={Proceedings of the ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
year = {2025},
}
```

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
