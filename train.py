
from ultralytics import YOLO

model = YOLO("yolov8m.pt")

model.train(
    data="data.yaml",   # update dataset path
    imgsz=640,
    epochs=50,
    batch=16
)
