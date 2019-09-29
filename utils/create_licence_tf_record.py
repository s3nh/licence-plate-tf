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