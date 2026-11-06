import sys
import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('cfg/AMG-YOLO.yaml')
    model.train(
                data=r'Datasets\HIT-UAV-Infrared-Thermal-Dataset-v1.2\HIT-UAV-v1.2\HIT-UAV-v1.2.yaml',  # HIT-UAV 数据集
                cache=False, # 如果内存充足可设为True加速训练
                imgsz=640,
                epochs=200,
                batch=16,
                workers=4,
                device='0',
                optimizer='SGD',
                lr0=0.01, # 初始学习率
                lrf=0.01, # 最终学习率 = lr0 * lrf = 0.0001

                project='runs/HIT-UAV',
                name='AMG-YOLO-train',
                )