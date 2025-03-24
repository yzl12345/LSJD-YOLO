import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
 
if __name__ == '__main__':
    model = YOLO('weights/best.pt') # select your model.pt path
    model.predict(source='datasets3/train/images',
                  imgsz=640,
                  project='runs/detect',
                  name='exp',
                  save=True,
                )

