
from __future__ import absolute_import  
from __future__ import division 
from __future__ import print_function

import hashlib

import io
import logging
import os 
import random
import re
import pandas as pd 

import contextlib2
from lxml import etree
import numpy as np 
from PIL import Image
from collections import namedtuple



import tensorflow as tf 

def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value = [value]))

def bytes_list_feature(value):
    return tf.train.Feature(bytes_list = tf.train.BytesList(value = value))


def float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))



def recursive_parse_txt_to_dict(txt):
    """
    Args:
    text: text file which contain id, boxes coordinate 
    
    Returns:
    Python dictionary holding TXT contents
    
    """
    pass


flags = tf.app.flags
flags.DEFINE_string('csv_input', '', 'Root directory to licence plate dataset')
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
    print("Jestem przed returnem !!!") 
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]

def create_tf_example(group, path):
    with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb')  as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size
    
    
    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmin = []
    ymin = []
    xmax = []
    ymax = []
    classes_text = []
    classes = []
    
    for index, row in group.object.iterrows():
        xmin.append(row['x_coord']/width)
        xmax.append( (row['x_coord'] + row['height'])/width) 
        ymin.append(row['y_coord']/height)
        ymax.append((row['y_coord'] + row['width'])/height)
        classes_text.append(row['licence'].encode('utf8'))
        classes.append(class_text_to_int(row['licence']))    
    tf_example = tf.train.Example(features = tf.train.Features(
        feature = {'image/height' : int64_feature(height), 
                   'image/width' : int64_feature(width),
                   'image/filename' : bytes_feature(filename),
                   'image/source_id': bytes_feature(filename), 
                   'image/encoded' : bytes_feature(encoded_jpg), 
                   'image/format' : bytes_feature(image_format), 
                   'image/object/bbox/xmin' : float_list_feature(xmin) ,
                   'image/object/bbox/ymin' : float_list_feature(ymin) , 
                   'image/object/bbox/xmax' : float_list_feature(xmax),
                   'image/object/bbox/ymax' : float_list_feature(ymax), 
                   'image/object/class/text' :   bytes_list_feature(classes_text), 
                   'image/object/class/label': int64_list_feature(classes) }
    ))
    return tf_example
    
def main(argv=None):
    writer = tf.python_io.TFRecordWriter(FLAGS.output_dir)
    path = os.path.join(FLAGS.image_dir)
    examples = pd.read_csv(FLAGS.csv_input)
    grouped = split(examples, 'filename')
    print("GROUPED")
    for group in grouped:
        tf_example = create_tf_example(group, path)
        writer.write(tf_example.SerializeToString())
        
    writer.close()
    output_path = os.path.join(os.getcwd(), FLAGS.output_dir)
    print("Succesfully created TFRecords {}".format(output_path))
    
    
if __name__ == "__main__":
    tf.compat.v1.app.run(main=main) 
        
        