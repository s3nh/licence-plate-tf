""" Object detection image using Tensorflow Classifier
"""


import os 
import cv2 
import numpy as np 
import tensorflow as tf 
import sys 

from utils import label_map_util
from utils import visualisation_utils as vis_util

MODEL_NAME = 'licence_detect'
CWD_PATH = os.getcwd()

PATH_TO_CKPT =  os.path.join(CWD_PATH, MODEL_NAME, 'frozen_inference_graph.pb')
PATH_TO_LABELS = os.path.join(CWD_PATH, 'training', 'labelmap.pbtxt')
NUM_CLASSES = 1


# Load tensorflow model 
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name = '')
    sess = tf.Session(graph=detection_graph)