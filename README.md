[Japanese/[English](https://github.com/Kazuhito00/YOLOX-Colaboratory-Training-Sample/blob/main/README_EN.md)] 

# YOLOX-Colaboratory-Training-Sample
<img src="https://user-images.githubusercontent.com/37477845/135489488-c55996d8-d32f-4612-8c99-8cdc37f7e7b2.gif" width="60%"><br>

[YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)をGoogle Colaboratory上で訓練しONNX、TensorFlow-Lite形式のファイルをエクスポートするサンプルです。<br>
以下の内容を含みます。<br>
* データセット ※アノテーション未実施
* データセット ※アノテーション済み
* Colaboratory用スクリプト(環境設定、モデル訓練)
* ONNX推論サンプル

# Requirement
* Pytorch 1.9.0 or later
* apex 0.1 or later
* pycocotools 2.0 or later
* OpenCV 3.4.2 or later ※推論サンプルを実施する場合のみ
* onnxruntime 1.5.2 or later ※推論サンプルを実施する場合のみ

# About annotation
[VoTT](https://github.com/microsoft/VoTT)を使用してアノテーションを行い、<br>
Pascal VOC形式で出力したアノテーションデータを前提としています。<br>
ただし、ノートブック内で更にMS COCO形式変換しています。<br><br>

ノートブックのサンプルでは、以下のようなディレクトリ構成を想定しています。<br>
ただし、本サンプルでは「pascal_label_map.pbtxt」は利用しないため、<Br>
格納しなくても問題ありません。
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
トレーニングはGoogle Colaboratory上で実施します。<br>
[Open In Colab]リンクからノートブックを開き、以下の順に実行してください。
1. YOLOX 依存パッケージインストール(YOLOX Dependent Package Install)
1. NVIDIA APEXインストール(NVIDIA APEX Install)
1. PyCocoToolsインストール(PyCocoTools Install)
1. データセットダウンロード(Download Dataset)<Br>自前のデータセットを使用したい方は「use_sample_image = True」をFalseに設定し、<br>「dataset_directory」に自前のデータセットのパスを指定してください
1. Pascal VOC形式 を MS COCO形式へ変換(Convert Pascal VOC format to MS COCO format)
1. モデル訓練(Training Model)<br>「!python train.py」を実施する前に「YOLOX」ディレクトリに「nano.py」を格納してください。<br>自前のデータセットを使用する場合「nano.py」の以下の項目を変更してください。
    1. クラス数<br>self.num_classes
    1. 画像格納パス<br>self.data_dir
    1. 学習データ アノテーションファイル<br>self.train_ann
    1. 検証データ アノテーションファイル<br>self.val_ann
    1. エポック数<br>self.max_epoch
1. 推論テスト(Inference test)
1. ONNX変換(Convert to ONNX)

※「nano.py」のオリジナルファイルは「[Megvii-BaseDetection/YOLOX/exps/default](https://github.com/Megvii-BaseDetection/YOLOX/tree/main/exps/default)」に格納されています

# Author
高橋かずひと(https://twitter.com/KzhtTkhs)
 
# License 
YOLOX-Colaboratory-Training-Sample is under [Apache-2.0 License](LICENSE).
