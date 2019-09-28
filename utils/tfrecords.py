import tensorflow as tf 
import numpy as np 
import glob
from PIL import Image

# Convert data into tfrecords format
# Int is used for numerical value 
# BYtes are used for character values
PATH = 'path'

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value = [value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


tfrecords_filename = 'licence_plate.tfrecords'
writer = tf.python_io.TFRecordWriter(tfrecords_filename)

images = glob.glob(PATH)
for image in images[:1]:
    img = Image.open(image)
    img = np.array(img.resize((32, 32, )))
    label = # READ TXT FILE HERE
    



example = tf.train.Example(features=tf.train.Features(feature=feature))


writer.write(example.SerializeToString()))
writer.close()
