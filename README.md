<<<<<<< HEAD
# LSJD-YOLO
=======
Install: git clone https://github.com/yzl12345/LSJD-YOLO
        cd LSJD-YOLO

1.初始训练：在mytrain.py文件中完成。
2.约束训练：在ultralytics/engine/trainer.py文件下，约396行，取消该部分的注释。完成后在mytrain.py文件中训练。
3.剪枝：在LL_pruning.py文件下可以调整剪枝率，对约束训练后的模型进行剪枝。
4.精度微调：在ultralytics/engine/trainer.py文件下，约590行，取消该部分的注释。完成后在mytrain.py文件中训练。
5.模型结果评价：把训练完成后的模型在val.py中进行测试。
6.推理加速：用model.export(format='ncnn', half=True)将模型导出为ncnn，在边缘计算设备上完成有效测试。
（论文实验中有效改进后的网络配置文件在MyModel文件夹下）
数据集：
论文数据集在datasets和datasets2文件夹下，一共有两套，都是水母单类别检测的数据集，含有大量水母小目标。
>>>>>>> 9d582b9 (first commit)
