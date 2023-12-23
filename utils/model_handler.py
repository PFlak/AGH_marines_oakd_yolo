import os.path
import re

default_model = "yolov8n.pt"


def extract_number(f):
    s = re.findall("\d+$", f)
    return int(s[0]) if s else -1, f


def best_model() -> str:
    if not os.path.isdir(os.path.join(os.getcwd(), "versions")):
        return default_model

    versions = os.listdir("versions")
    max_version = max(versions, key=extract_number)

    return os.path.join(os.getcwd(), "versions", max_version, "weights", "best.pt")

