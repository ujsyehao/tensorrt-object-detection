# tensorrt-object-detection

## tutuorial
This project supports the conversion of caffe model to tensorrt model.
It includes common IPlugin layer implementation and common errors in conversion.

## support detection model
* SSD
* mobilenet-ssd
* mobilenetv2-ssd
* Refinedet

## support multiple resolution
* 320x320
* 512x512
* 300x300

## support tensorrt version
* tensorrt 4.0.0.4
* tensorrt 4.0.1.6
* tensorrt 5.0.2.6

## Iplugin layer description
* detection_output(refinedet used)
* relu6

## compile the project
* download the link[https://pan.baidu.com/s/170phX6-q7kj5y9DmFC1yug], copy the caffemodel to model directory
* modify CMakeLists.txt(cuda path, opencv path, tensorrt path)
* run a.sh

## TODO
* update code
* add multiple version support
