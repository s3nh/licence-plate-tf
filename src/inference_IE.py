import time
import cv2 
import logging as log 
import time
from openvino.inference_engine import IENetwork, IEPlugin

model_xml = 'tf-mobilenet/training/frozen_inference_graph.xml'
model_bin = 'tf-mobilenet/training/frozen_inference_graph.bin'
image_path = 'tf-mobilenet/valid_.jpg'
image = cv2.imread(image_path)
image = cv2.resize(image, (600, 600))
print(image.size)

print("Start infer loading time {}".format(time.time()))
now = time.time()
blob = cv2.dnn.blobFromImage(image)
plugin = IEPlugin(device="MYRIAD", plugin_dirs=None)
net = IENetwork(model = model_xml, weights = model_bin)
n, c, h, w = net.inputs['image_tensor'].shape
out_blob = net.outputs
print(out_blob)
print("TIme passed {}".format(time.time() - now))
# Loading model to the plugin 



print("Start infer loading time {}".format(time.time()))
infer_time = time.time()
exec_net = plugin.load(network = net) 
print("inference time  passed {}".format(time.time() - infer_time))

pred_time = time.time()
res = exec_net.infer(inputs = {'image_info' : [h, w, 1], 
                               'image_tensor' : blob 
                               })
print("Prediction time passed {}".format(time.time() - pred_time))
res = res['detection_output'][0][0][0]
print(res)
xmin = int(res[3] * 600)
xmax = int(res[5] * 600)
ymin = int(res[4] * 600)
ymax = int(res[6] * 600)

print("xmin {} xmax {} ymin {} ymax {}".format(xmin, xmax, ymin, ymax))

licence = image[ymin:ymax, xmin:xmax]
cv2.imwrite('licence.jpg', licence )
