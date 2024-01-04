import os
import threading
import uuid
import scipy.sparse
import shutil
from os import path
import cv2
import numpy as np
from multiprocessing.pool import ThreadPool as Pool
from scipy.sparse.linalg import spsolve
import time

BACKGROUND_DIR = "backgrounds"
INPUT_DIR = "datasets/valid"
POOL_SIZE = 8

backgrounds = []


def laplacian_matrix(n, m):
    mat_D = scipy.sparse.lil_matrix((m, m))
    mat_D.setdiag(-1, -1)
    mat_D.setdiag(4)
    mat_D.setdiag(-1, 1)

    mat_A = scipy.sparse.block_diag([mat_D] * n).tolil()

    mat_A.setdiag(-1, 1 * m)
    mat_A.setdiag(-1, -1 * m)

    return mat_A


def load_background(width, height):
    files = [f for f in os.listdir(BACKGROUND_DIR)]
    for file in files:
        img = cv2.imread(path.join(BACKGROUND_DIR, file))
        backgrounds.append(cv2.resize(img, (width, height)))


def create_mask(width, height, boxes):
    img = np.zeros((width, height, 3), dtype=np.uint8)
    for box in boxes:
        img = cv2.rectangle(img, (int(box[0] * 640 - box[2] * 640 / 2), int(box[1] * 640 - box[3] * 640 / 2)),
                            (int(box[0] * 640 + box[2] * 640 / 2), int(box[1] * 640 + box[3] * 640 / 2)),
                            (255, 255, 255), -1)
    return img


def get_boxes(source_name):
    boxes = []
    name = os.path.splitext(source_name)[0]
    with open(path.join(INPUT_DIR, "labels", name + ".txt"), "r") as f:
        for line in f.readlines():
            boxes.append(list(map(float, line.strip().split(" ")[1:])))
    return boxes


def source_elements(source_name):
    source = cv2.imread(path.join(INPUT_DIR, "images", source_name))
    boxes = get_boxes(source_name)
    mask = create_mask(source.shape[0], source.shape[1], boxes)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    M = np.float32([[1, 0, 0], [0, 1, 0]])

    y_max, x_max = source.shape[:-1]
    y_min, x_min = 0, 0
    x_range = x_max - x_min
    y_range = y_max - y_min

    source = cv2.warpAffine(source, M, (x_range, y_range))

    mask = mask[y_min:y_max, x_min:x_max]
    mask[mask != 0] = 1

    mat_A = laplacian_matrix(y_range, x_range)
    laplacian = mat_A.tocsc()
    for y in range(1, y_range - 1):
        for x in range(1, x_range - 1):
            if mask[y, x] == 0:
                k = x + y * x_range
                mat_A[k, k] = 1
                mat_A[k, k + 1] = 0
                mat_A[k, k - 1] = 0
                mat_A[k, k + x_range] = 0
                mat_A[k, k - x_range] = 0
    mat_A = mat_A.tocsc()
    mask_flat = mask.flatten()
    return source, mat_A, mask_flat, laplacian


def edit(source_name, source, mat_A, mask_flat, laplacian, target):
    start_time = time.perf_counter()
    target = target.copy()

    y_max, x_max = target.shape[:-1]
    y_min, x_min = 0, 0
    x_range = x_max - x_min
    y_range = y_max - y_min

    print("2.", time.perf_counter() - start_time)
    start_time = time.perf_counter()
    for channel in range(source.shape[2]):
        source_flat = source[y_min:y_max, x_min:x_max, channel].flatten()
        target_flat = target[y_min:y_max, x_min:x_max, channel].flatten()

        # inside the mask:
        # \Delta f = div v = \Delta g
        alpha = 1
        mat_b = laplacian.dot(source_flat) * alpha

        # outside the mask:
        # f = t
        mat_b[mask_flat == 0] = target_flat[mask_flat == 0]

        x = spsolve(mat_A, mat_b)
        x = x.reshape((y_range, x_range))
        x[x > 255] = 255
        x[x < 0] = 0
        x = x.astype('uint8')

        target[y_min:y_max, x_min:x_max, channel] = x
    print("3.", time.perf_counter() - start_time)
    name = uuid.uuid4()
    cv2.imwrite(path.join(INPUT_DIR, "images", str(name) + ".jpg"), target)

    old_name = os.path.splitext(source_name)[0]
    shutil.copyfile(path.join(INPUT_DIR, "labels", str(old_name) + ".txt"),
                    path.join(INPUT_DIR, "labels", str(name) + ".txt"))


load_background(640, 640)

files = [f for f in os.listdir(os.path.join(INPUT_DIR, "images"))]
i = 0
lock_a = threading.Lock()


def worker(file):
    global i
    start_time = time.perf_counter()
    source, mat_A, mask_flat, laplacian = source_elements(file)
    # with lock_a:
    print(file, " mask create", time.perf_counter() - start_time)
    start_time = time.perf_counter()
    for background in backgrounds:
        edit(file, source, mat_A, mask_flat, laplacian, background)
        # with lock_a:
        print(i, len(backgrounds) * len(files), int(i / (len(backgrounds) * len(files)) * 100))
        i += 1
    # with lock_a:
    print("complete", time.perf_counter() - start_time)


a = time.perf_counter()
# pool = Pool(POOL_SIZE)

print("START")
for file in files:
    # pool.apply_async(worker, (file,))
    worker(file)
# pool.close()
# pool.join()
print("END", time.perf_counter() - a)
