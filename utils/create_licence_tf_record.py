import hashlib
import io
import logging
import os 
import random
import re


import contextlib2
from lxml import etree
import numpy as np 
from PIL import Image


from __future__ import absolute_import  
from __future__ import division 
from __future__ import print_function


import tensorflow as tf 




def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value = [value]))


def bytes_list_feature(value):
    return tf.train.Feature(bytes_list = tf.train.BytesList(value = value))


def recursive parse_txt_to_dict(txt):
    """
    
    Args:
    text: text file which contain id, boxes coordinate 
    
    Returns:
    Python dictionary holding TXT contents
    
    """
    
    
    
    pass



flags = tf.app.flags
flags.DEFINE_string('data_dir', '', 'Root directory to licence plate dataset')
flags.DEFINE_string('output_dir', '', 'Path to directory to output TFRecords')
flags.DEFINE_string('label_map_path', 'data/licence_label_map.pbtxt')
FLAGS = flags.FLAGS

tf.logging.set_verbosity(tf.logging.INFO)


def get_class_name_from_filename(file_name):
    """
    It is not needed for 1class detector challenge
    """
    pass


def create_tf_example(image, 
                      annotation_list, 
                      image_dir 
                     ):
    
    """ Converts image and labels to tf.Example proto
    
    Args:
        image: 
    
    """
