import math
import os
import cv2
import pandas as pd
from matplotlib import pyplot as plt
from ultralytics import YOLO
from utils.model_handler import best_model

x_line = 10
conf_limit = 0.5
input_dir = "./datasets/test"
out_dir = "./out/"
count = 0
countBadClass = 0
if not os.path.isdir(out_dir):
    os.mkdir(out_dir)

data = {
    "class": [],
    "conf": [],
    "xDiff": [],
    "yDiff": [],
    "widthDiff": [],
    "heightDiff": [],
}

files = [f for f in os.listdir(os.path.join(input_dir, "images"))]
model = YOLO(best_model("versions"))

for file in files:
    count += 1
    name = os.path.splitext(file)[0]
    img = cv2.imread(os.path.join(input_dir, "images", file))
    results = model.predict(img)

    correct_data = {}

    with open(os.path.join(input_dir, "labels", name + ".txt"), "r") as f:
        for line in f.readlines():
            temp = list(map(float, line.strip().split(" ")))
            correct_data[int(temp[0])] = temp[1:]
    print(correct_data)

    for r in results:
        for box in r.boxes:
            b = box.xyxy[0]
            if b[1] > x_line and box.conf.item() > conf_limit:
                c = box.cls
                startpos = (int(b[0].item()), int(b[1].item()))
                endpos = (int(b[2].item()), int(b[3].item()))
                # cv2.rectangle(img, startpos, endpos, (255, 0, 0), 2)
                # cv2.putText(img, r.names[c.item()] + " " + str(math.floor(box.conf.item() * 100)) + "%",
                #             (startpos[0], startpos[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                #             (36, 255, 12),
                #             2)
                if c.item() in correct_data:

                    data['class'].append(c.item())
                    data["conf"].append(box.conf.item())
                    data["xDiff"].append(abs((startpos[0] + endpos[0]) / (2 * 640) - correct_data[c.item()][0]))
                    data["yDiff"].append(abs((startpos[1] + endpos[1]) / (2 * 640) - correct_data[c.item()][1]))
                    data["widthDiff"].append(abs((endpos[0] - startpos[0]) / 640 - correct_data[c.item()][2]))
                    data["heightDiff"].append(abs((endpos[1] - startpos[1]) / 640 - correct_data[c.item()][3]))
                    print(r.names[c.item()], startpos, endpos, correct_data[c.item()])
                else:
                    print("NOT FOUND")
                    countBadClass += 1

print(data)
df = pd.DataFrame(data)
print(df)
df.to_csv('out.csv', index=False)
print(len(df['class']), countBadClass, count)

# df['conf'].hist()
# plt.show()
