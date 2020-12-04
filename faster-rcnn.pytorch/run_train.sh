#!/usr/bin/env bash

# GPU_ID=0,1,2,3
# DATASET="gta_det"
GPU_ID=0
DATASET="carla"
NET="res101"
BATCH_SIZE=1
WORKER_NUMBER=4
LEARNING_RATE=1e-3
DECAY_STEP=10
SESSION=888
EPOCH=20

LOADSESSION=666
LOADEPOCH=5
LOADCHECKPOINT=5300
RESUME=1

# If you are training with pretrain model 'faster_rcnn_200_14_18895.pth', uncomment the last three line

CUDA_VISIBLE_DEVICES=${GPU_ID} python trainval_net.py \
    --dataset ${DATASET} \
    --net ${NET} \
    --s ${SESSION} \
    --bs ${BATCH_SIZE} \
    --nw ${WORKER_NUMBER} \
    --lr ${LEARNING_RATE} \
    --lr_decay_step ${DECAY_STEP} \
    --cuda \
    --mGPUs \
    --epochs ${EPOCH} \
    --o adam \
    # --r ${RESUME}\
    # --checksession ${LOADSESSION} \
    # --checkepoch ${LOADEPOCH} \
    # --checkpoint ${LOADCHECKPOINT} 

echo $(whoami)" : "${SESSION}_${LOADEPOCH}_${LOADCHECKPOINT}" Finish!"
