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
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.group.keys(), gb.groups)]

def create_tf_example(group, path):
    with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb')  as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size
    
    
    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    x = []
    y = []
    height = []
    width = []
    classes = []
    
    for index, row in group.object.iterrows():
        x.append(row['x'])
        height.append(row['height']) 
        y.append(row['y'])
        width.append(row['width'])
        classes.append(row['class'].encode('utf8'))
        
    tf_example = tf.train.Example(features = tf.train.Features(
        feature = {'image/height' : int64_feature(height), 
                   'image/width' : int64_feature(width),
                   'image/filename' : bytes_feature(filename), 
                   'image/filename' : bytes_feature(encoded_jpg), 
                   'image/format' : bytes_feature(image_format), 
                   'image/object/bbox/xmin' : ,
                   'image/object/bbox/ymin' : , 
                   'image/object/bbox/xmax' : ,
                   'image/object/bbox/ymax' : , 
                   'image/object/class/text' :  
                
                   }
    ))
    return tf_example

    
    
        
    
def main():
    writer = tf.python_io.TFRecordWriter(FLAGS.output_dir)
    path = os.path.join(FLAGS.image_dir)
    examples = pd.read_csv(FLAGS.csv_input)
    grouped = split(examples, 'filename')
    for group in grouped:
        tf_example = create_tf_example(group, path)
        writer.write(tf_example.SerializeToString())
        
    writer.close()
    output_path = os.path.join(os.getcwd(), FLAGS.output_path)
    print("Succesfully created TFRecords {}".format(output_path))
    
    
if __name__ == "__main__":
    tf.app.run() 
        
        