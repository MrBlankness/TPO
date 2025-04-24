# TPO (Tree Preference Optimization)

Welcome to the official GitHub repository for TPO (Tree Preference Optimization)!

This is the official code for paper: **TPO: Aligning Large Language Models with Multi-branch & Multi-step Preference Trees**

**📢 News: this work has been accepted at the ICLR 2025 !**

**If you find our project interesting or helpful, we would appreciate it if you could give us a star! Your support is a tremendous encouragement to us!**

## Environment

- 1+ A100(80G)

## Install Environment

We use conda to manage the environment.
Please refer to the following steps to install the environment:

```sh
conda create -n TPO python=3.11 -y
conda activate TPO
pip install -r requirements.txt
```

## Data Preparation

- Step 1: Download data from the official [Step-DPO](https://github.com/dvlab-research/Step-DPO) library
- Step 2: Use GPT-4 to generate tree data and score each response
```sh
python data_prepare.py
```

## Train Model
```sh
python tpo_train.py
```

## Model Evaluation
- see eval/run.sh

## Publication

```
@inproceedings{liao2024tpo,
  title={TPO: Aligning Large Language Models with Multi-branch \& Multi-step Preference Trees},
  author={Liao, Weibin and Chu, Xu and Wang, Yasha},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2025}
}
```
