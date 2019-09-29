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
flags.DEFINE_string('image_dir', '', 'Path to images')
FLAGS = flags.FLAGS

tf.logging.set_verbosity(tf.logging.INFO)

def class_text_to_int(row_label):
    if row_label == 'licence':
        return 1
    else:
        None
        
        
def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb=df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.group.keys(), gb.groups)]


def create_tf_example(group, path):
    with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb')  as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size
    
    
    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes = []
    classes_text = []
    
    
    

        