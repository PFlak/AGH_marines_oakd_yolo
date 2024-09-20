import time
import uuid
from os import path

from ultralytics import YOLO
import cv2
import math

from utils.model_handler import best_model

TIME_SAVE = 2
OUTPUT_DIR = "./out_camera/"
last_save_photo = time.time()

# start webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 640)

while True:
    success, img = cap.read()

    cv2.imshow('Webcam', img)
    # print(time.time(), last_save_photo)
    if time.time() - last_save_photo > TIME_SAVE:
        print("PHOTO")
        last_save_photo = time.time()
        name = uuid.uuid4()
        cv2.imwrite(path.join(OUTPUT_DIR, str(name) + ".jpg"), img)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
