[[Japanese](https://github.com/Kazuhito00/YOLOX-Colaboratory-Training-Sample)/English] 

# YOLOX-Colaboratory-Training-Sample
<img src="https://user-images.githubusercontent.com/37477845/135489488-c55996d8-d32f-4612-8c99-8cdc37f7e7b2.gif" width="60%"><br>

This is a sample to train [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) on Google Colaboratory and export a file in ONNX format and TensorFlow-Lite format.<br>
It includes the following contents.<br>
* Data set(Annotation not implemented)
* Data set(Annotated)
* Colaboratory script (environment setting, model training)
* ONNX inference sample

# Requirement
* Pytorch 1.9.0 or later
* apex 0.1 or later
* pycocotools 2.0 or later
* OpenCV 3.4.2 or later
* onnxruntime 1.5.2 or later ※Only when performing inference samples

# About annotation
It is assumed that annotation data is annotated using VoTT and output in Pascal VOC format.<br>
However, it is further converted to MS COCO format in the notebook.<br><br>

The notebook sample assumes the following directory structure.<br>
However, since "pascal_label_map.pbtxt" is not used in this sample, <Br>
There is no problem even if you do not store it.
```
02.annotation_data
│  000001.jpg
│  000001.xml
│  000002.jpg
│  000002.xml
│   :
│  000049.jpg
│  000049.xml
│  000050.xml
└─ pascal_label_map.pbtxt
  
```

# Usage
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Kazuhito00/YOLOX-Colaboratory-Training-Sample/blob/main/YOLOX_Colaboratory_Training_Sample.ipynb)<br>
Training will be conducted on Google Colaboratory.<br>
Open your notebook from the [Open In Colab] link and run it in the following order:
1. YOLOX 依存パッケージインストール(YOLOX Dependent Package Install)
1. NVIDIA APEXインストール(NVIDIA APEX Install)
1. PyCocoToolsインストール(PyCocoTools Install)
1. データセットダウンロード(Download Dataset)<Br>If you want to use your own dataset, set "use_sample_image = True" to False and specify the path of your own dataset in <br> "dataset_directory".
1. Pascal VOC形式 を MS COCO形式へ変換(Convert Pascal VOC format to MS COCO format)
1. モデル訓練(Training Model)<br>Please store "ano.py" in the "YOLOX" directory before executing "!python train.py". <br>When using your own data set, change the following items in "**nanodet-m.yml**".
    1. Number of classes<br>self.num_classes
    1. Image storage path<br>self.data_dir
    1. Training data annotation file<br>self.train_ann
    1. Validation data annotation file<br>self.val_ann
    1. Number of epochs<br>self.max_epoch
1. 推論テスト(Inference test)
1. ONNX変換(Convert to ONNX)
  
※The original file of "nano.py" is stored in "[Megvii-BaseDetection/YOLOX/exps/default](https://github.com/Megvii-BaseDetection/YOLOX/tree/main/exps/default)"

# Author
Kazuhito Takahashi(https://twitter.com/KzhtTkhs)
 
# License 
YOLOX-Colaboratory-Training-Sample is under [Apache-2.0 License](LICENSE).
