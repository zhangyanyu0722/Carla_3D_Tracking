# Monocular 3D Vehicle Detection and Tracking in CARLA

<p align="center">
  <img src="https://github.com/zhangyanyu0722/Carla_Tracking/blob/main/image/2D.gif" height="250" width="400"/>
  <img src="https://github.com/zhangyanyu0722/Carla_Tracking/blob/main/image/3D.gif" height="250" width="400"/>
</p>

## Introduction

- We present a mixed convolutional neural network to playing the CarRacing-v0 using imitation learning in OpenAI Gym. After training, this model can automatically detect the boundaries of road features and drive the robot like a human.

## Prerequisites
```diff
! NOTE: this repo is made for PyTorch 1.6 compatible issue, the generated results might be changed.
```
- Linux (tested on Ubuntu 16.04.4 LTS)
- Python 3.6.9
    - `3.6.9` tested
- PyTorch 1.6.0
    - `1.6.0` (with CUDA 10.2, torchvision 0.7.0)
- Anaconda

## Quick Start
- In this section, you will train a model from scratch, test our pretrained models, and reproduce our evaluation results.

### Installation
- Clone this repo:
```bash
git clone https://github.com/zhangyanyu0722/Car_Racing_DL.git
cd Car_Racing_DL/
```

- Install PyTorch 1.6.0 and torchvision 0.7.0 from http://pytorch.org and other dependencies.
```bash
# Install gym (recommand conda)
conda install -c conda-forge gym
# Or
pip install 'gym[all]'
```
- Install requirements.
```bash
pip3 install -r requirements.txt
```
### Data Preparation (Optical)

- For a quick start, we suggest downing this [dataset](https://drive.google.com/file/d/1RtoSgk78raI549A3BE8uYRV9Ly9DY7GJ/view?usp=sharing) and drop it into ```Car_Racing_DL``` folder.
```diff
! NOTE: Click "Download Anyway" if it shows Google Drive can't scan this file for viruses.
```
- Download and unzip [dataset](https://drive.google.com/file/d/1RtoSgk78raI549A3BE8uYRV9Ly9DY7GJ/view?usp=sharing)
```bash
# make a folder under repo.
mkdir data
# Unzip the dataset.
tar -xvf teacher.tar.gz -C data
```
- Data Preprocessing : we map all actions into 7 classes and randomly delete 50% dataset in the "Accelerate" class.
```bash
# Data Preprocessing
python3 preprocessing.py
```

### Train Model (Optical)
```bash
# Train the model, do not recommand if do not have a GPU
python3 main.py train
```

### Model Evaluation

- For convenience, we provide many pre-trained models here, just click to download.

Channel | Model | Score | Model | Score | Model | Score | Model | Score
-----|------|------|------|------|------|------|------|------
RGB | [VGG16_RGB](https://drive.google.com/file/d/1GLK9af4OUU8GmmNMiOh61pmKWKhCqWmH/view?usp=sharing) | 438.8 | [AlexNet_RGB](https://drive.google.com/file/d/17L2ZqE12jmdBLMrPzQEOWPDvcDAU8q9h/view?usp=sharing) | 471.5 | [EasyNet_RGB](https://drive.google.com/file/d/1npkvXvTZvkxhyx7EIRlzGEc5L8U5I_r3/view?usp=sharing) | 594.9
Gray | [VGG16_Gray](https://drive.google.com/file/d/1a63waR8AA-yNFJ8FjUKkhu0cvYXjb7VU/view?usp=sharing) | 432.4 | [AlexNet_Gray](https://drive.google.com/file/d/17n-Zf5HyKIYqP9Vh95UrbIYXIblEz8a4/view?usp=sharing) | 464.9 | [EasyNet_Gray](https://drive.google.com/file/d/1xsCawTvq3nVlHreO2e8IqXa7XzhoxL7L/view?usp=sharing) | 558.3 | [Best Model](https://drive.google.com/file/d/1krJQd033m6JzdB2LUGMqO0YKjGhPiMRj/view?usp=sharing) | 649.1

- Then you need to rename it as ```train.t7``` and put it under ```Car_Racing_DL/data```
- Test the final performance to control a robot in Gym
```bash
# Test the model
python3 main.py test

# Score the model
python3 main.py score
```

## LICENSE
See [LICENSE](https://github.com/zhangyanyu0722/Carla_Tracking/blob/master/LICENSE) for details. Third-party datasets and tools are subject to their respective licenses.
