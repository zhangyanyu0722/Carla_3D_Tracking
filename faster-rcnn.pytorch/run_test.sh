#!/usr/bin/env bash

GPU_ID=0
PHASE="val" # "test", "train"
DATASET="carla"
NET="res101"
BATCH_SIZE=1
WORKER_NUMBER=4
SESSION=666
TEST_EPOCH=20
TEST_CHECKPOINT=19462

CUDA_VISIBLE_DEVICES=${GPU_ID} python test_net.py \
    --dataset ${DATASET} \
    --net ${NET} \
    --checksession ${SESSION} \
    --checkepoch ${TEST_EPOCH} \
    --checkpoint ${TEST_CHECKPOINT} \
    --anno ${PHASE} \
    --cuda \
    # --vis

echo 
echo $(whoami)" : "${SESSION}_${TEST_EPOCH}_${TEST_CHECKPOINT}" Finish!"
echo "Copy output pkl file to dataset folder..."
cp \
    vis/faster_rcnn_${SESSION}_${TEST_EPOCH}_${TEST_CHECKPOINT}/detections_${PHASE}.pkl \
    data/carla/carla_${PHASE}_detections.pkl
echo "Done!!"
