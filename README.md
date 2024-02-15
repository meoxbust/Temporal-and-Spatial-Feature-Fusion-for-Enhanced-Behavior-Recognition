# Temporal-and-Spatial-Feature-Fusion-for-Enhanced-Behavior-Recognition
This research focuses on enhancing the capability to recognize students’ behaviors in a classroom 
environment by combining temporal and spatial feature. The proposed model consists of two main branches
called SwinT-D-CNN-LSTM. First, branch utilizes a combination of the Swin Transformer (SwinT) model 
and Deep Convolutional Neural Network (D-CNN) to process RGB image data, thereby enhancing the ability 
to detect the smallest details and complex spatial relationships within the image. Then, branch employs a 
Skeleton - Long Short-Term Memory (SkeLSTM) model to track temporal skeleton data, capturing changes 
in student postures and movements over time. Finally, combination SwinT-D-CNN and SkeLSTM feature to 
classify behavior. Additionally, we build a dataset comprising 4 behaviors – Reading, Writing, Sleeping, and 
Raising Hand – was constructed to facilitate comparison and evaluation of the proposed method.

![Fusion architecture](https://github.com/meoxbust/Temporal-and-Spatial-Feature-Fusion-for-Enhanced-Behavior-Recognition/blob/main/assets/fusion.png)
# Project Organization
```
├── LICENSE
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── Feature Extraction      <- The processing of extracting features from raw data.
│   │   ├── core        <- Apply Yolov8-pose estimation for detecting a person's action.   
│   │   ├── main.py 
│   │      
│   └── raw videos          <- The original, all raw videos. 
│
├── functions               <- Define general functions used for training models.
│
├── models             <- All implemented model structures applied in this project.
│   ├── D-CNN      <- Deep-convolutional neural networks.
│   ├── LSTM       <- Long Short-Term Memory.
│   ├── ResNet18   <- Residual Networks.
│   ├── Swin-Transformer    <- Hierarchical Vision Transformer using Shifted Windows.
│
│
├── plots          <- Plot RGB images in data 
│  
└── src    <- The data picked for testing, training
```
