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
    
    net = cv.dnn.readNet('training/frozen_inference_graph.xml',
                        'training/frozen_inference_graph.bin'
                        )

    # Specify target device 

    #net.setPreferableTarget(cv.dnn.CPU)
    # Read an image 
    frame = cv.imread('dataset/benchmarks/endtoend/eu/eu1.jpg')
    frame = cv.resize(frame, (600,600))
    print(frame.shape)
    #frame =  np.expand_dims(frame, axis = 0)
    if frame is None:
        raise Exception('Image not Foune')
    blob = cv.dnn.blobFromImage(frame, 0.007843137718737125)
    net.setInput(blob)
    out = net.forward() 
    
if __name__ == "__main__":
    main()