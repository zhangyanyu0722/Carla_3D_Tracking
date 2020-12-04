# --------------------------------------------------------
# Fast/er R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------


import os.path as osp
import numpy as np
import scipy.sparse
import json
import cv2
import os

from glob import glob

from datasets.imdb import imdb
import datasets.ds_utils as ds
# from model.utils.config import cfg

# Use visualization or not
VISUALIZATION = False


class Carla(imdb):
    """
    Load information of the CARLA dataset
    and preprocess them
    """
    def __init__(self, dataPath, split):
        self.dataPath = dataPath
        self.split = split

        # Set classes and their mapping
        # classNames = ["Background", "Car", "Pedestrian"]
        classNames = ["Background", "Car"]
        self.cls2Ind = {c: i for i, c in enumerate(classNames)}

        # 3x4, Camera matrix
        # A mapping between the 3d world and a 2d image
        # The camera is at (960.0, 540.0), focal length f = 960.0000000000001
        FOCAL_LENGTH = 960.0000000000001
        self.CAM_CALIBRATION = np.array([
            [FOCAL_LENGTH, 0.0, 960.0, 0.0],
            [0.0, FOCAL_LENGTH, 540.0, 0.0],
            [0.0, 0.0, 1.0, 0.0]
        ])

        # Phase validation
        assert split in ["train", "val", "test"]

        # Constructor call
        super().__init__("carla_" + split, classes = classNames)

        # 1. Get data path of the current split
        self.splitPath = self._getSplitPath()
        # 2. Read infos and names of image
        self.infos, self.endvids, self.imgNames = self._readInfos()
        # 3. Set image index
        self._image_index = np.arange(len(self.infos)).astype(int)
        # 4. Set roidb handler (see imdb.py for details)
        self.set_proposal_method("gt")

    # Private methods
    def _getSplitPath(self):
        return osp.join(self.dataPath, self.split)

    def _readInfos(self):
        pattern = osp.join(self.splitPath, "label", "*.json")
        labelPathFiles = sorted(glob(pattern))

        infos = []
        imgNames = []
        endvids = []

        # Read all files storing all label paths
        for labelPathFile in labelPathFiles:
            with open(labelPathFile, "r") as f:
                # labelPaths = f.readlines()
                labelPaths = json.loads(f.read()) 
                numFrames = len(labelPaths)
                endvids += ([False] * numFrames)
                endvids[-1] = True # flag of the last frame

                for rawLabelPath in labelPaths:
                    labelPath = rawLabelPath.strip("\"\n")
                    with open(labelPath, "r") as lf:
                        info = json.load(lf)
                        infos.append(info)
                        # Get the image name
                        imgNames.append(info["name"])
        
        return infos, endvids, np.array(imgNames)

    def _projection(self, locs):
        """
        Convert from 3d to 2d

        3d              - [x, y, z, 1]
        2d              - [x, y]
        camera location - [p_x, p_y]

        relation: 
        x_2d = (x_3d * f / z_3d) + p_x
        y_2d = (y_3d * f/ z_3d) + p_y

        Params:

        `locs`: Bx3, 3d Center of a batch of objects
        """
        # [X, Y, Z, 1]
        extendLocs = np.c_[locs, np.ones((len(locs), 1))]
        # [f * X + p_x * Z, f * Y + p_y * Z, Z]
        projLocs = extendLocs @ self.CAM_CALIBRATION.T 
        # Reduced by Z: [f * X / Z + p_x, f * Y / Z + p_y]
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
    
    def _drawBBoxAndCenter(self, image, boxes, centers):
        """
        Draw a bounding box with a projected 3d center
        """
        color = (0, 0, 204)

        for i in range(len(centers)):
            bbox = tuple(boxes[i].astype(int))
            center = tuple(centers[i].astype(int))
            # print(bbox, center, type(bbox), type(center))
            
            cv2.rectangle(image, bbox[0:2], bbox[2:4], color, 2)
            cv2.circle(image, center, 4, color, -1)
        
        return image


    # For gt_roidb method
    def _loadCarlaAnnotation(self, index, vis = False):
        """
        Load the annotations of the ith image from the info list
        """
        width, height = 1920, 1080

        # "labels"
        info = self.infos[self.image_id_at(index)]
        labels = info["labels"]

        # Object ids
        objIds = ds.get_label_array(labels, 
                                    key_list = ["id"], 
                                    empty_shape = (0,)).astype(int)
        nObjects = len(objIds)

        # Ignores
        ignores = ds.get_label_array(labels, 
                                    key_list = ["attributes", "ignore"],
                                    empty_shape = (0,)).astype(int)

        # Ground-truth class
        objClasses = ds.getClassArray(labels, self.classes[0], emptyShape = (0,)) # Class name of objects
        gtClassses = np.array([self.cls2Ind[c] for c in objClasses]) # convert to the index of class
        
        # Boxes
        boxes = ds.get_box2d_array(labels)[:, :4] # ignore the confidence

        # 3d object centers
        locations = ds.get_label_array(labels,
                                        key_list = ["box3d", "location"],
                                        empty_shape = (0, 3)).astype(float)
        centers = self._projection(locations)
        
        # Overlaps (one-hot vector form of labels)
        overlaps = np.zeros((nObjects, self.num_classes), dtype = float)
        overlaps[np.arange(nObjects), gtClassses] += 1

        overlaps = scipy.sparse.csr_matrix(overlaps) # convert to a sparse matrix

        # Area of segments
        segAreas = self._computeSegAreas(boxes)

        # End-of-video id (flag)
        endvids = np.zeros(nObjects, dtype = int)
        if (self.endvids[self.image_id_at(index)]):
            # Last image of the video, set them to 1
            endvids += 1

        # Validation boxes
        flag = ds.validate_boxes(boxes, width=width, height=height)

        # Information set
        infoSet = None

        if (flag):
            # Debug
            # print(self.image_path_at(index))
            # print(endvids)
            
            if vis:
                im = cv2.imread(self.image_path_at(index))
                im2show = np.copy(im)
                im2show = self._drawBBoxAndCenter(im2show, boxes, centers)
                cv2.imwrite(os.path.join("vis", "inputs", 'result%d.png' % (index + 1)),
                            im2show)

            infoSet = {
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

        return infoSet

    # imdb class methods
    def image_path_at(self, i):
        """
        Return the absolute path to the image i
        """
        imgName = self.imgNames[self.image_id_at(i)]
        imgPath = osp.join(self.splitPath, "image", imgName)

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
        infoSets = []
        newImgInd = []

        # Debug
        delItems = []

        for i in self._image_index:
            infoSet = self._loadCarlaAnnotation(i, vis = VISUALIZATION)
            if (infoSet):
                infoSets.append(infoSet)
                newImgInd.append(i)
            else:
                fileNum = self.image_path_at(i).split("/")[-1].split(".")[0]
                delItems.append(fileNum)
        
        print(delItems)
        print(len(delItems))

        # Filter out indices of invalid images
        self._image_index = newImgInd.copy()

        return infoSets

