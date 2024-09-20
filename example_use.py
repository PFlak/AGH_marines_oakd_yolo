import math
import os
import cv2
from ultralytics import YOLO
from utils.model_handler import best_model

x_line = 10
conf_limit = 0.50
input_dir = "./datasets/train/"
out_dir = "./out/"
if not os.path.isdir(out_dir):
    os.mkdir(out_dir)

files = [f for f in os.listdir(os.path.join(input_dir, "images"))]
model = YOLO(best_model("versions"))
for file in files:
    img = cv2.imread(os.path.join(input_dir, "images", file))
    results = model.predict(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    for r in results:
        for box in r.boxes:
            b = box.xyxy[0]
            if b[1].item() > x_line and box.conf.item() > conf_limit:
                c = box.cls
                startpos = (int(b[0].item()), int(b[1].item()))
                endpos = (int(b[2].item()), int(b[3].item()))
                cv2.rectangle(img, startpos, endpos, (255, 0, 0), 2)
                cv2.putText(img, r.names[c.item()] + " " + str(math.floor(box.conf.item() * 100)) + "%",
                            (startpos[0], startpos[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                            (36, 255, 12),
                            2)
    print(os.path.join(out_dir, file))
    cv2.imwrite(os.path.join(out_dir, file), img)
