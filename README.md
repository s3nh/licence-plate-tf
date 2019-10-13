## Tensorflow-Lite object detection 
Transfer Learning . 

This repo is just an simple version of 

[Tensorflow transfer learning on TPU](https://medium.com/tensorflow/training-and-serving-a-realtime-mobile-object-detector-in-30-minutes-with-cloud-tpus-b78971cf1193)

### Deploy

Running on Intel Compute stick 2 


#### Installation 

as mention in [documentation](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md), 

installation requires as follows:

###
- Protobuf 3.0.0
- Python-tk
- Pillow 1.0
- lxml
-tf Slim (which is included in the "tensorflow/models/research/" checkout)
- Jupyter notebook
- Matplotlib
- Tensorflow (>=1.12.0)
- Cython
- contextlib2
- cocoapi


#### Run training

add python path
export PYTHONPATH=$PYTHONPATH=/home/s3nh/tf-mobilenet/models/research:/home/s3nh/tf-mobilenet/models/research/slim

Create tf records


python utils/create_licence_tf_record.py --csv_input=dataset/benchmarks/endtoend/labels.csv  --output_dir=training/train.record --image_dir=dataset/benchmarks/endtoend/eu


train, based on pipeline.config file and training files. 
python3 models/research/object_detection/model_main.py  --logtostderr --train_dir=training/ --pipeline_config_path=training/pipeline.config

#### Results after 4000 steps 


![results](./images/detection_boxes_precision.PNG)



#### Predictions 


![results](./images/boxes_.PNG)