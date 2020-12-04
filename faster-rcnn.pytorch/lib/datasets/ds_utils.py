# --------------------------------------------------------
# Fast/er R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from functools import reduce  # forward compatibility for Python 3
import operator

import numpy as np


def unique_boxes(boxes, scale=1.0):
    """Return indices of unique boxes."""
    v = np.array([1, 1e3, 1e6, 1e9])
    hashes = np.round(boxes * scale).dot(v)
    _, index = np.unique(hashes, return_index=True)
    return np.sort(index)


def xywh_to_xyxy(boxes):
    """Convert [x y w h] box format to [x1 y1 x2 y2] format."""
    return np.hstack((boxes[:, 0:2], boxes[:, 0:2] + boxes[:, 2:4] - 1))


def xyxy_to_xywh(boxes):
    """Convert [x1 y1 x2 y2] box format to [x y w h] format."""
    return np.hstack((boxes[:, 0:2], boxes[:, 2:4] - boxes[:, 0:2] + 1))


def validate_boxes(boxes, width=0, height=0):
    """Check that a set of boxes are valid."""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    flag1 = (x1 >= 0).all()
    flag2 = (y1 >= 0).all()
    flag3 = (x2 >= x1).all()
    flag4 = (y2 >= y1).all()
    flag5 = (x2 < width).all()
    flag6 = (y2 < height).all()

    return flag1 and flag2 and flag3 \
            and flag4 and flag5 and flag6


def filter_small_boxes(boxes, min_size):
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    keep = np.where((w >= min_size) & (h > min_size))[0]
    return keep

# GTA data utilities
def get_from_dict(data_dict, map_list):
    assert isinstance(map_list, list)
    return reduce(operator.getitem, map_list, data_dict)


def get_label_array(boxes, key_list, empty_shape):
    if len(boxes) == 0:
        return np.empty(empty_shape)
    return np.array([get_from_dict(box, key_list) for box in boxes])


def get_box2d_array(boxes):
    if len(boxes) == 0:
        return np.empty([0, 5])
    return np.array([[box['box2d']['x1'],
                      box['box2d']['y1'],
                      box['box2d']['x2'],
                      box['box2d']['y2'],
                      box['box2d']['confidence']] for box in boxes],
                    dtype=np.float32)

# Carla data utilities

def getClassArray(labels, defaultClass, emptyShape):
    classes = get_label_array(labels,
                            key_list = ["category"],
                            empty_shape = emptyShape)

    if (classes.shape == emptyShape):
        return np.array([defaultClass])
    
    return classes
    
def get2dBoxes(objList):
    """
    Get 2d box of objects
    Hard-code part:
    1. [4, 8)-th entries correspond to x1, y1, x2, y2
    """
    if (len(objList) == 0):
        return np.empty((0, 4))

    return objList[:, 4:8].astype(float) 

def get3dLocations(objList):
    """
    Get the 3d location of the center in camera's view
    Hard-code part:
    1. [11, 14)-th entries correspond to 3d location (x, y, z)
    """
    if (len(objList) == 0):
        return np.empty((0, 3))

    return objList[:, 11:14].astype(float)