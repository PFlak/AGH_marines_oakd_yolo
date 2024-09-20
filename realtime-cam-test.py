import numpy as np
from ultralytics import YOLO
import cv2
import math

from utils.model_handler import best_model

# start webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 640)

cols = 3
ratio = 640 / 640
width = 900

# model
model = YOLO(best_model("versions"))


def process_frame(img):
    results = model(img, stream=True)

    # coordinates
    for r in results:
        boxes = r.boxes

        for box in boxes:
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # convert to int values

            # put box in cam
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # confidence
            confidence = math.ceil((box.conf[0] * 100)) / 100
            print("Confidence --->", confidence)

            # class name
            cls = int(box.cls[0])
            print("Class name -->", r.names[cls])

            # object details
            org = [x1, y1]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2

            cv2.putText(img, r.names[cls], org, font, fontScale, color, thickness)
    return img


def make_window(*imgs):
    temp_imgs = []
    for i in range(len(imgs)):
        temp_imgs.append(cv2.resize(imgs[i], (int(width // cols), int(width // cols * ratio))))

    result = None
    for row in range(math.ceil(len(temp_imgs) / cols)):
        r = temp_imgs[row * cols]
        for col in range(1, cols):
            if row * cols + col >= len(temp_imgs):
                img = np.zeros(temp_imgs[0].shape, dtype=np.uint8)
                r = cv2.hconcat([r, img])
            else:
                r = cv2.hconcat([r, temp_imgs[row * cols + col]])
        if result is None:
            result = r
        else:
            result = cv2.vconcat([result, r])

    return result


def only_blue(img):
    img[:, :, 1] = 0
    img[:, :, 2] = 0
    return img


def only_green(img):
    img[:, :, 0] = 0
    img[:, :, 2] = 0
    return img


def only_red(img):
    img[:, :, 0] = 0
    img[:, :, 1] = 0
    return img


while True:
    success, img = cap.read()
    img1 = process_frame(img.copy())
    img2 = process_frame(only_blue(img.copy()))
    img3 = process_frame(only_red(img.copy()))
    img4 = process_frame(only_green(img.copy()))
    img5 = process_frame(cv2.rotate(img.copy(), cv2.ROTATE_90_CLOCKWISE))
    img6 = process_frame(cv2.rotate(img.copy(), cv2.ROTATE_180))
    img7 = process_frame(cv2.rotate(img.copy(), cv2.ROTATE_90_COUNTERCLOCKWISE))
    img8 = process_frame(cv2.flip(img.copy(), 0))
    img9 = process_frame(cv2.flip(img.copy(), 1))
    cv2.imshow('Webcam', make_window(img1, img2, img3, img4, img5, img6, img7, img8, img9))
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
