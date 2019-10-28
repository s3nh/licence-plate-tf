import cv2 
import logging as log 
from time import time 
from openvino.inference_engine import IENetwork, IEPlugin

model_xml = 'tf-mobilenet/training/frozen_inference_graph.xml'
model_bin = 'tf-mobilenet/training/frozen_inference_graph.bin'
image_path = 'tf-mobilenet/valid_.jpg'
image = cv2.imread(image_path)
image = cv2.resize(image, (600, 600))
print(image.size)

blob = cv2.dnn.blobFromImage(image)
plugin = IEPlugin(device="MYRIAD", plugin_dirs=None)
net = IENetwork(model = model_xml, weights = model_bin)
n, c, h, w = net.inputs['image_tensor'].shape
out_blob = net.outputs
print(out_blob)
# Loading model to the plugin 


exec_net = plugin.load(network = net) 
res = exec_net.infer(inputs = {'image_info' : [h, w, 1], 
                               'image_tensor' : blob 
                               })


res = res['detection_output']
print(res)
