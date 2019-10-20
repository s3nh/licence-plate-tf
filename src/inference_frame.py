"""
Inference 
compiled model 
on one frame shot, to check if it is possible to run 
model on cv.dnn.Device
"""

# Imports 

import cv2 as cv
import numpy as np 

# Read Compile net 
def main():
    
    net = cv.dnn.readNet('pretrain/frozen_inference_graph.xml',
                        'pretrain/frozen_inference_graph.bin'
                        )

    # Specify target device 

    net.setPreferableTarget(cv.dnn.DNN_TARGET_MYRIAD)
    print(net)
    # Read an image 
    frame = cv.imread('../dataset/benchmarks/endtoend/eu/eu1.jpg')
    #frame =  np.expand_dims(frame, axis = 0)
    if frame is None:
        raise Exception('Image not Foune')

    #NExt step/ 
    # Blob preparation and perform an inference
    
    blob = cv.dnn.blobFromImage(cv.resize(frame, (600, 600)))
    net.setInput(blob)
    out = net.forward() 
    
if __name__ == "__main__":
    main()