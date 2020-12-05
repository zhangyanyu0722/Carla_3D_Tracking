# Monocular 3D Vehicle Detection and Tracking in CARLA

<p align="center">
  <img src="https://github.com/zhangyanyu0722/Carla_3D_Tracking/blob/main/image/2D.gif" height="250" width="400"/>
  <img src="https://github.com/zhangyanyu0722/Carla_3D_Tracking/blob/main/image/3D.gif" height="250" width="400"/>
</p>

## Introduction

- We present a novel framework that jointly detects and tracks 3D vehicle bounding boxes. Our approach leverages 3D pose estimation to learn 2D patch association overtime and uses temporal information from tracking to obtain stable 3D estimation.

## Prerequisites
```diff
! NOTE: this repo is made for PyTorch 1.0+ compatible issue, the generated results might be changed.
```

- Linux (tested on Ubuntu 16.04.4 LTS)
- Python 3.6.9
    - `3.6.4` tested
    - `3.6.9` tested
- PyTorch 1.3.1 
    - `1.0.0` (with CUDA 9.0, torchvision 0.2.1)
    - `1.1.0` (with CUDA 9.0, torchvision 0.3.0)
    - `1.3.1` (with CUDA 10.1, torchvision 0.4.2)
- nvcc 10.1
    - `9.0.176`, `10.1` compiling and execution tested
    - `9.2.88` execution only
- gcc 5.4.0
- Pyenv or Anaconda

and Python dependencies list in `3d-tracking/requirements.txt` 

## Quick Start
In this section, you will train a model from scratch, test our pretrained models, and reproduce our evaluation results.
For more detailed instructions, please refer to [`DOCUMENTATION.md`](3d-tracking/DOCUMENTATION.md).

### Installation
- Clone this repo:
```bash
git clone https://github.com/zhangyanyu0722/Carla_3D_Tracking.git
cd Carla_3D_Tracking/
```

- Install PyTorch 1.0.0+ and torchvision from http://pytorch.org and other dependencies. You can create a virtual environment by the following:
```bash
# Add path to bashrc 
echo -e '\nexport PYENV_ROOT="$HOME/.pyenv"\nexport PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo -e 'if command -v pyenv 1>/dev/null 2>&1; then\n  eval "$(pyenv init -)"\nfi' >> ~/.bashrc

# Install pyenv
curl -L https://raw.githubusercontent.com/pyenv/pyenv-installer/master/bin/pyenv-installer | bash

# Restart a new terminal if "exec $SHELL" doesn't work
exec $SHELL

# Install and activate Python in pyenv
pyenv install 3.6.9
pyenv local 3.6.9
```

- Install requirements, create folders and compile binaries for detection
```bash
cd 3d-tracking
bash scripts/init.sh
cd ..

cd faster-rcnn.pytorch
bash init.sh
```

> NOTE: For [faster-rcnn-pytorch](faster-rcnn.pytorch/lib/setup.py) compiling problems 
[[1](https://github.com/jwyang/faster-rcnn.pytorch/issues/410#issuecomment-450709668)], please compile COCO API and replace pycocotools.

> NOTE: For [object-ap-eval](https://github.com/traveller59/kitti-object-eval-python#dependencies) compiling problem. It only supports python 3.6+, need `numpy`, `skimage`, `numba`, `fire`. If you have Anaconda, just install `cudatoolkit` in anaconda. Otherwise, please reference to this [page](https://github.com/numba/numba#custom-python-environments) to set up llvm and cuda for numba.

### Data Preparation

- Download and extract CARLA 0.8.4 from https://github.com/carla-simulator/carla/releases/tag/0.8.4  
This project expects the carla folder to be inside this project i.e PythonClient/carla-data-export/carla  
Install all the necessary requirements for your python environment using:
```
pip install -r requirements.txt
```

- Before the data generation scripts can be run you must start a CARLA server. This can be done by running the executable in the CARLA root folder with the appropriate parameters. Running the server on windows in a small 200x200 window would for example be:
```
./CarlaUE4.exe -carla-server -fps=10 -windowed -ResX=200 -ResY=200
```
- Once the server is running, data generation can be started using (remove --autopilot for manual control):
```
python datageneration.py --autopilot
```
- Download the [pretrained model](https://drive.google.com/file/d/1UtC_Kn_qaUZQ3IobGkLkIPXL6wukklo6/view?usp=sharing)

### Execution

For running a whole pipeline (2D proposals, 3D estimation and tracking):
```bash
# Generate predicted bounding boxes for object proposals
cd faster-rcnn.pytorch/

# Step 00 (Optional) - Training on CARLA dataset
./run_train.sh

# Step 01 - Generate bounding boxes
./run_test.sh
```

```bash
# Given object proposal bounding boxes and 3D center from faster-rcnn.pytorch directory
cd 3d-tracking/

# Step 00 - Data Preprocessing
# Collect features into json files (check variables in the code)
python loader/gen_pred.py carla val

# Step 01 - 3D Estimation
# Running single task scripts mentioned below and training by yourself
# or alternatively, using multi-GPUs and multi-processes to run through all sequences
python run_estimation.py carla val --session 888 --epoch 030

# Step 02 - 3D Tracking and Evaluation
# 3D helps tracking part. For tracking evaluation, 
# using multi-GPUs and multi-processes to run through all sequences
python run_tracking.py carla val --session 888 --epoch 030

# Step 03 - 3D AP Evaluation
# Plot the 2D/3D/Birdview figure
python tools/plot_tracking.py carla val --session 888 --epoch 030
```

> Note: If facing `ModuleNotFoundError: No module named 'utils'` problem, please add `PYTHONPATH=.` before `python {scripts} {arguments}`.


## LICENSE
See [LICENSE](https://github.com/zhangyanyu0722/Carla_Tracking/blob/master/LICENSE) for details. Third-party datasets and tools are subject to their respective licenses.

## Acknowledgements
We thank [faster.rcnn.pytorch](https://github.com/jwyang/faster-rcnn.pytorch) for the detection codebase, [pymot](https://github.com/Videmo/pymot) for their MOT evaluation tool and [kitti-object-eval-python](https://github.com/traveller59/kitti-object-eval-python) for the 3D AP calculation tool.
