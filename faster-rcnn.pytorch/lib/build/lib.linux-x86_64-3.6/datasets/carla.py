# --------------------------------------------------------
# Fast/er R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------


import os.path as osp
import numpy as np
import scipy.sparse
import json

from glob import glob

from datasets.imdb import imdb
import datasets.ds_utils as ds
from model.utils.config import cfg

class Carla(imdb):
    """
    Load information of the CARLA dataset
    and preprocess them
    """
    def __init__(self, dataPath, split):
        self.dataPath = dataPath
        self.split = split

        # Set classes and their mapping
        classNames = ["Background", "Car", "Pedestrian"]
        self.cls2Ind = {c: i for i, c in enumerate(classNames)}

        # Calibration matrix of the camera
        self.CAM_CALIBRATION = np.array([
            [935.3074360871937, 0, 960, 0],
            [0, 935.3074360871937, 540, 0],
            [0, 0, 1, 0]
        ])

        # Phase validation check
        assert split in ["train", "val", "test"]

        # Constructor call
        super().__init__("carla_" + split, classes = classNames)

        # 1. Get data path of the current split
        self.splitPath = self._getSplitPath()
        # 2. Read infos and names of image
        self.infos, self.imgNames = self._readInfos()
        # print(self.imgNames)
        # 3. Set image index
        self._image_index = np.arange(len(self.infos)).astype(int)
        # 4. Set roidb handler (see imdb.py for details)
        self.set_proposal_method("gt")

    # Private methods
    def _getSplitPath(self):
        return osp.join(self.dataPath, self.split)

    def _readInfos(self):
        pattern = osp.join(self.splitPath, "label", "*.txt")
        labelFiles = sorted(glob(pattern))

        infos = []
        imgNames = []

        print(self.splitPath)
        if not labelFiles:
            raise RuntimeError("Label files not found")

        for name in labelFiles:
            with open(name, "r") as lf:
                # Read info of all objects and store them into 
                # a list of lists
                objList = []
                for i, line in enumerate(lf.readlines()):
                    line = line.strip()
                    if line:
                        if (i == 0):
                            imgNames.append(line)
                            continue

                        objInfo = line.split()
                        objList.append(objInfo)

                infos.append(objList)

        return infos, np.array(imgNames)

    # TODO: calibration
    def _convertTo(self, locs):
        """
        Coordinate conversion from camera's view 
        to the ?'s view

        Params:

        `locs`: Bx3, Center coordinate of a batch of objects
        """
        extendLocs = np.c_[locs, np.ones((len(locs), 1))]
        projLocs = extendLocs @ self.CAM_CALIBRATION.T 
        
        # Normalization
        projLocs = projLocs[:, :2] / projLocs[:, 2:3]

        return projLocs
    
    def _computeSegAreas(self, boxes):
        """
        Compute the area of boxes
        """
        x1s = boxes[:, 0]
        y1s = boxes[:, 1]
        x2s = boxes[:, 2]
        y2s = boxes[:, 3]

        segAreas = (x2s - x1s + 1) * (y2s - y1s + 1)

        return segAreas

    # For gt_roidb method
    def _loadCarlaAnnotation(self, index):
        """
        Load the annotations of the ith image from the info list
        """
        width, height = 1920, 1080

        objList = np.array(self.infos[self.image_id_at(index)])
        nObjects = len(objList)

        # Ignores
        ignores = np.array([False] * nObjects, dtype = int)

        # Ground-truth class
        objClasses = ds.getClasses(objList, self.classes[0]) # Class name of objects
        gtClassses = np.array([self.cls2Ind[c] for c in objClasses]) # convert to the index of class
        
        # Boxes
        boxes = ds.get2dBoxes(objList)

        # 3d object centers
        locations = ds.get3dLocations(objList)
        centers = self._convertTo(locations)

        # Overlaps (one-hot vector form of labels)
        overlaps = np.zeros((nObjects, self.num_classes), dtype = float)
        overlaps[np.arange(nObjects), gtClassses] += 1
        overlaps = scipy.sparse.csr_matrix(overlaps) # convert to a sparse matrix

        # Area of segments
        segAreas = self._computeSegAreas(boxes)

        # End-of-video id (flag)
        endvids = np.zeros(nObjects, dtype = int)
        if (index == len(self.infos) - 1):
            # Last image of the video, set them to 1
            endvids += 1

        # Validation boxes
        ds.validate_boxes(boxes, width=width, height=height)

        info_set = {
            "width": width,
            "height": height,
            "boxes": boxes,
            "gt_classes": gtClassses,
            "gt_overlaps": overlaps,
            "flipped": False,
            "seg_areas": segAreas,
            "ignore": ignores,
            "end_vid": endvids,
            "center": centers
            }

        return info_set

    # imdb class methods
    def image_path_at(self, i):
        """
        Return the absolute path to the image i
        """
        imgName = self.imgNames[self.image_id_at(i)]
        imgPath = osp.join(self.splitPath, "image", imgName)

        # print("Path:\n")
        # print(imgPath)

        return imgPath

    def image_id_at(self, i):
        """
        Return the id of image i (hard-coded as i)
        """
        return i

    # For roidb handler
    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        Returns:

        List of annotations (required on imdb.py) on each image
        """
        return [self._loadCarlaAnnotation(i)
                for i in self._image_index]

