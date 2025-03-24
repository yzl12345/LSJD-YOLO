import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('weights/best.pt')
    model.val(data=r'cfg_yaml/mycoco.yaml',
              split='test',#val
              imgsz=640,
              batch=32,
              # rect=False,
              save_json=True, # 这个保存coco精度指标的开关
              project='runs/val',
              name='exp',
              )