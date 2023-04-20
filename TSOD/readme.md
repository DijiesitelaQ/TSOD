# yolox-SPPST
This repo is the implementation of ["Pre-locator Incorporating Swin-Transformer Refined Classifier for Traffic Sign Recognition"].
this paper proposes a novel two-stage traffic sign detection method consisting of a pre-locator network, which directly acquires sub-regions that may contain traffic signs from the original image, and a refined classification network, which performs the traffic sign refinement recognition task in the sub-regions; in order to more effectively learn the special spatial information of traffic sign presence, an innovative SPP-ST module is proposed, which combines the Spatial Pyramid Pool module (SPP) is integrated with the Swin-Transformer module as a new feature extractor.

# Install
```bash
$ git clone https://github.com/DijiesitelaQ/TSOD
$ cd TSOD
$ pip install -r requirements.txt
```
<summary>Install</summary>

[**Python>=3.7.0**](https://www.python.org/) is required with all
[requirements.txt](https://github.com/ultralytics/yolov5/blob/master/requirements.txt) installed including
[**PyTorch>=1.7**](https://pytorch.org/get-started/locally/):
<!-- $ sudo apt update && apt install -y libgl1-mesa-glx libsm6 libxext6 libxrender-dev -->
# Inference

* `Datasets` : [TT100k](http://cg.cs.tsinghua.edu.cn/traffic-sign/)
* `preweights` : [`yolox-SPPST.pt`](https://github.com/DijiesitelaQ/TSOD/weights/yolox-sppst.pt)


```bash
$ python detect.py --source inference/images/ --conf 0.25 --img-size 640 

```
# Train
train.py allows you to train new model from strach.
```bash
$ python train.py --batch-size 32 --img 640 640 --data 218class.yaml --cfg yolox-sppst.yaml --weights weights/yolox-sppst.pt --device 0  --hyp hyp.scratch.yolox --epochs 300

```
See the [YOLOv5 Docs](https://docs.ultralytics.com) for full documentation on training, testing and deployment.
```
```

# References
Thanks to their great works
* [yolox](https://github.com/Megvii-BaseDetection/YOLOX)
* [ultralytics/yolov5](https://github.com/ultralytics/yolov5)
