
# LSJD-YOLO
These are the source codes for our paper titled "Optimized LSJD-YOLO: Lightweight Small Jellyfish Detection Using Enhanced YOLOv11 for Underwater Edge Computing" published at the Visual Computer journal in 2022.
=======
Install: git clone https://github.com/yzl12345/LSJD-YOLO
        cd LSJD-YOLO

The improvements to the network primarily focus on the neck and detection heads:

1.A 1×1 convolutional layer has been added to the neck to compress the number of channels, thereby achieving network lightweighting.
2.An additional detection head has been incorporated on top of the existing architecture to address the challenge of detecting small objects.

head:
  - [-1, 1, Conv, [256, 1, 1]]                          # 11
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [6, 1, Conv, [256, 1, 1, None, 1, 1, False]]        # 13 input_proj.1
  - [[-2, -1], 1, Concat, [1]]                          # cat backbone P4
  - [-1, 2, C3k2, [256, False]]                         # 15

  - [-1, 1, Conv, [256, 1, 1]]                          # 16
  - [-1, 1, nn.Upsample, [None, 2, 'nearest']]
  - [4, 1, Conv, [256, 1, 1, None, 1, 1, False]]        # 18 input_proj.0
  - [[-2, -1], 1, Concat, [1]]                          # cat backbone P3
  - [-1, 2, C3k2, [256, False]]                         # 20 (P3/8-small)

  - [-1, 1, Conv, [128, 1, 1]]                          # 21（可以改成128对比效果）
  - [-1, 1, nn.Upsample, [None, 2, 'nearest']]
  - [2, 1, Conv, [128, 1, 1, None, 1, 1, False]]        # 23 input_proj.0
  - [[-2, -1], 1, Concat, [1]]                          # cat backbone P2
  - [-1, 2, C3k2, [128, False]]                         # 25 (P2/4-xsmall)

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 21], 1, Concat, [1]]                          # cat head P3
  - [-1, 2, C3k2, [256, False]]                         # 28 (P3/8-small)

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 16], 1, Concat, [1]]                          # cat head P4
  - [-1, 2, C3k2, [256, False]]                         # 31 (P4/16-medium)

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 11], 1, Concat, [1]]                          # cat head P5
  - [-1, 2, C3k2, [256, True]]                          # 34 (P5/32-large)

  - [[25, 28, 31, 34], 1, Detect, [nc]]                 # Detect(P2, P3, P4, P5)

Below are the training steps:

1.Initial Training: This step is carried out in the mytrain.py file.
2.Constrained Training: In the ultralytics/engine/trainer.py file, locate approximately line 396 and uncomment the relevant section. After making this modification, proceed with training in the mytrain.py file.
3.Pruning: The pruning rate can be adjusted in the LL_pruning.py file. Apply pruning to the model that has undergone constrained training.
4.Fine-Tuning for Accuracy: In the ultralytics/engine/trainer.py file, locate approximately line 590 and uncomment the relevant section. Afterward, continue training in the mytrain.py file.
5.Model Evaluation: Test the trained model using the val.py script to evaluate its performance.
6.Inference Acceleration: Export the model to the NCNN format using model.export(format='ncnn', half=True). This allows for efficient testing on edge computing devices.

The results of the ablation experiments and comparative experiments on the two datasets are presented as follows:

![0K4D120P$`FPL(SEAAC@2VC](https://github.com/user-attachments/assets/d06a7213-b8f6-409e-afe3-34a48abecfad)

![BZ$C@SGB7)(R6IDS}2B)56H](https://github.com/user-attachments/assets/43e6c6b1-d12f-4e50-90eb-7a12651cfba0)




(The configuration file of the effectively improved network used in the paper's experiments is located in the "MyModel" folder.)

Datasets:
The datasets used in the paper are stored in the "datasets" and "datasets2" folders. There are two sets of datasets in total, both of which are dedicated to single-category detection of jellyfish and contain a large number of small jellyfish targets.
>>>>>>> 9d582b9 (first commit)
