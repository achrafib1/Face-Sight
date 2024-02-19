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
    """
    Loads a pre-trained model from the given path.

    Parameters:
    - model_path: The path to the pre-trained model.

    Returns:
    - model: The loaded model.
    - names: The names of the classes that the model can predict.
    - scale_coords: A function to scale the coordinates of the bounding boxes to the original image size.
    - non_max_suppression: A function to apply non-maximum suppression to the model's predictions.
    - plot_one_box: A function to draw a bounding box on the image.
    """

    model = attempt_load(model_path)
    names = model.module.names if hasattr(model, "module") else model.names
    return model, names, scale_coords, non_max_suppression, plot_one_box
