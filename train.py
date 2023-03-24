from ultralytics import YOLO
model = YOLO("./pre_model/yolov8s.pt")
model.train(data="custom.yaml", batch=8, imgsz=640, epochs=100, workera=1)