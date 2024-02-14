# Temporal-and-Spatial-Feature-Fusion-for-Enhanced-Behavior-Recognition
# Data Organization
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
