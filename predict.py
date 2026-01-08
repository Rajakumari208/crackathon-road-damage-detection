from ultralytics import YOLO

model = YOLO("yolov8m_rdd.pt")

model.predict(
    source="test/images",   # update test image path
    imgsz=640,
    conf=0.35,
    save_txt=True,
    save_conf=True
)
