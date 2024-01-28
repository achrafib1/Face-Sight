import cv2
from yolov7.models.experimental import attempt_load
from yolov7.utils.general import scale_coords, non_max_suppression
from yolov7.utils.plots import plot_one_box


def load_model(model_path):
    model = attempt_load(model_path)
    names = model.module.names if hasattr(model, "module") else model.names
    return model, names, scale_coords, non_max_suppression, plot_one_box
