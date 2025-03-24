from ultralytics import YOLO
import os
import time

start_time = time.time()            #开始时间

## 配置文件路径
cfg_path = 'cfg_yaml/my.yaml'
coco_path = 'cfg_yaml/mycoco.yaml'


# 1.初始训练（先调文件ultralytics/engine/trainer.py中的参数！！！！！！！！！！！！！！！！！！！！！！！！！）
def step1_train():
    model = YOLO(cfg_path)
    model.train(data=coco_path, imgsz=640, epochs=300, batch=16)  # train the model
    model.export(format='onnx', simplify=True, dynamic=False, imgsz=640)
    model.val(data=r'cfg_yaml/mycoco.yaml', split='test', imgsz=640, batch=32, save_json=True, # 这个保存coco精度指标的开关
              project='runs/val', name='exp')

# 2.约束训练（调文件ultralytics/engine/trainer.py中的剪枝参数！！！！！！！！！！！！！！！！！！！！！！！！！）
def step2_Constraint_train():
    model = YOLO(cfg_path)
    model.train(data=coco_path, imgsz=640, epochs=120, batch=16, amp=False)  # train the model
    model.export(format='onnx', simplify=True, dynamic=False, imgsz=640)
    model.val(data=r'cfg_yaml/mycoco.yaml', split='test', imgsz=640, batch=32, save_json=True, # 这个保存coco精度指标的开关
              project='runs/val', name='exp')

# 3.剪枝
def step3_pruning():
    from LL_pruning import do_pruning
    do_pruning("weights/best.pt", "weights/prune.pt")
    model = YOLO("weights/prune.pt")
    model.export(format='onnx', simplify=True, dynamic=False, imgsz=640)
    model.val(data=r'cfg_yaml/mycoco.yaml', split='test', imgsz=640, batch=32, save_json=True, # 这个保存coco精度指标的开关
              project='runs/val', name='exp')

# 4.精度回调（需要调文件ultralytics/engine/trainer.py中的参数！！！！！！！！！！！！！！！！！！！！！！）
def step4_finetune():
    model = YOLO("weights/prune.pt")     # load a pretrained model (recommended for training)
    model.train(data=coco_path, imgsz=640, epochs=180, batch=32)  # train the model
    model.export(format='onnx', simplify=True, dynamic=False, imgsz=640)
    model.val(data=r'cfg_yaml/mycoco.yaml', split='test', imgsz=640, batch=32, save_json=True, # 这个保存coco精度指标的开关
              project='runs/val', name='exp')
    print(model.info(detailed=True))

# 查看模型结构
def show_model():
    model = YOLO("weights/prune.pt")     # load a pretrained model (recommended for training)
    # 查看模型的结构（包括参数量和计算量）
    print(model.info(detailed=True))
    # model.export(format='onnx', simplify=True, dynamic=False, imgsz=640)
    model.export(format='ncnn', half=True)

step1_train()
# step2_Constraint_train()
# step3_pruning()
# step4_finetune()
# show_model()

end_time = time.time()          #结束时间
print('Running time: %s Seconds' % (end_time - start_time))



