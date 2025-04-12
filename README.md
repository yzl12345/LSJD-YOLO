
# LSJD-YOLO
=======
Install: git clone https://github.com/yzl12345/LSJD-YOLO
        cd LSJD-YOLO

对网络的改进，主要集中在颈部和检测头：
1.在颈部增加了1*1卷积层压缩通道数，实现网络轻量化。
2.在原有基础上增加一个检测头，解决小目标难以检测的问题。

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

如下是训练步骤：
1.初始训练：在mytrain.py文件中完成。
2.约束训练：在ultralytics/engine/trainer.py文件下，约396行，取消该部分的注释。完成后在mytrain.py文件中训练。
3.剪枝：在LL_pruning.py文件下可以调整剪枝率，对约束训练后的模型进行剪枝。
4.精度微调：在ultralytics/engine/trainer.py文件下，约590行，取消该部分的注释。完成后在mytrain.py文件中训练。
5.模型结果评价：把训练完成后的模型在val.py中进行测试。
6.推理加速：用model.export(format='ncnn', half=True)将模型导出为ncnn，在边缘计算设备上完成有效测试。

在两个数据集上的消融实验和对比实验结果分别如下：

![0K4D120P$`FPL(SEAAC@2VC](https://github.com/user-attachments/assets/d06a7213-b8f6-409e-afe3-34a48abecfad)

![BZ$C@SGB7)(R6IDS}2B)56H](https://github.com/user-attachments/assets/43e6c6b1-d12f-4e50-90eb-7a12651cfba0)




（论文实验中有效改进后的网络配置文件在MyModel文件夹下）
数据集：
论文数据集在datasets和datasets2文件夹下，一共有两套，都是水母单类别检测的数据集，含有大量水母小目标。
>>>>>>> 9d582b9 (first commit)
