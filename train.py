from ultralytics import YOLO
import torch
#import torch
print(torch.cuda.is_available())
#torch.cuda.set_device(0)

if __name__ == '__main__':
    # Load model
    model = YOLO("last.pt") #YOLO("yolov8n.yaml") # model from scratch

    # Use model
    results = model.train(data="data.yaml", epochs=100, batch=6, save_period=10) # train model