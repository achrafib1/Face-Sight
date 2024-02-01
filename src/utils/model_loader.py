import cv2
import os
import sys

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../yolov7"))
)
from models.experimental import attempt_load
from utils.general import scale_coords, non_max_suppression
from utils.plots import plot_one_box


def load_model(model_path):
    model = attempt_load(model_path)
    names = model.module.names if hasattr(model, "module") else model.names
    return model, names, scale_coords, non_max_suppression, plot_one_box
