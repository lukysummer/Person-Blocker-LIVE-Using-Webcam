# Live Person Blocker Using Webcam

<p align="center"><img src="example_image.png" height = "400"></p>

Block out all persons live (as shown in **Black Mirror's White Christmas episode**), using webcam and google colab notebook (with GPU)

See [Colab Notebook](https://github.com/lukysummer/Person-Blocker-Using-Webcam/blob/master/PersonBlocker_Webcam.ipynb) for full instructions.

## Components

* Object detection using facebook's [Detectron2](https://github.com/facebookresearch/detectron2/blob/master/detectron2/utils/visualizer.py) (in PyTorch + takes only 0.1 to 0.2 seconds for a single image frame's object detection + visualization using a pre-trained model)
* Person Blocker Visualization following [this repo](https://github.com/minimaxir/person-blocker) (didn't use its object detection backbone, since it takes 20 to 30 seconds per frame & has conflicts with newest version of Keras and Tensorflow)
* Webcam embedding in Google Colab Notebook following this [notebook](https://github.com/vindruid/yolov3-in-colab/blob/master/yolov3_streaming_webcam.ipynb) and [medium post](https://towardsdatascience.com/yolov3-pytorch-streaming-on-google-colab-16f4fe72e7b)

## Next Steps

* Connect phone camera to computer and block people in the phone video
* Decrease delay in the live video blocker
		  
## Sources

* https://github.com/facebookresearch/detectron2
* https://github.com/minimaxir/person-blocker 
* https://github.com/vindruid/yolov3-in-colab/blob/master/yolov3_streaming_webcam.ipynb
* https://towardsdatascience.com/yolov3-pytorch-streaming-on-google-colab-16f4fe72e7b
