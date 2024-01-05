import os.path
import re

default_model = "yolov8n.pt"


def extract_number(f):
    s = re.findall("\d+$", f)
    return int(s[0]) if s else -1, f


def best_model(dir_name) -> str:
    if not os.path.isdir(os.path.join(os.getcwd(), dir_name)):
        return default_model

    versions = os.listdir(os.path.join(os.getcwd(), dir_name))
    max_version = max(versions, key=extract_number)

    return os.path.join(os.getcwd(), dir_name, max_version, "weights", "best.pt")
