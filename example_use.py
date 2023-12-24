import os

import cv2
from ultralytics import YOLO

from utils.model_handler import best_model

x_line = 10

files = [f for f in os.listdir("./datasets/valid/images")]
model = YOLO(best_model())
for file in files:
    img = cv2.imread(os.path.join("./datasets/valid/images", file))
    results = model.predict(img)
    for r in results:
        for box in r.boxes:
            b = box.xyxy[0]
            if b[1] > x_line:
                c = box.cls
                startpos = (int(b[0].item()), int(b[1].item()))
                endpos = (int(b[2].item()), int(b[3].item()))
                print(b)
                print(startpos, endpos)
                cv2.rectangle(img, startpos, endpos, (255, 0, 0), 2)
    cv2.imwrite(os.path.join("./out/", file), img)
