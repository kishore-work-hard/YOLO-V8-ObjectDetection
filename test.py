from ultralytics import YOLO
import cv2
import time

model = YOLO("./pre_model/yolov8s.pt")
cam_url = 'car.jpg'
while True:
    t1 = time.time()
    video = cv2.VideoCapture(cam_url)
    # video = cv2.VideoCapture(0)
    ret, frame = video.read()
    if ret:
        x = model.predict(source=frame, show=True, classes=[2, 3, 7])
        # print(x)
        for box in x[0].boxes:
            print(box.numpy())
