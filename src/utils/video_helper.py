from streamlit_webrtc import VideoTransformerBase
import av
from .image_processor import process_image
from .box_drawer import draw_boxes
from .predict import predict
import cv2
import numpy as np


def process_frame(
    frame,
    model,
    names,
    faces,
    non_max_suppression,
    scale_coords,
    plot_one_box,
    strategies,
    background,
):
    """
    Processes a frame for object detection and applies various strategies.

    Parameters:
    - frame: The original frame.
    - model: The object detection model to use.
    - names: The names of the classes.
    - faces: A list to store the detected faces.
    - non_max_suppression: The non-maximum suppression function to apply to the model's predictions.
    - scale_coords: The function to rescale the coordinates to the original image size.
    - plot_one_box: The function to draw a bounding box on the image.
    - strategies: A list of strategies to apply to the image.
    - background: The desired background color in hexadecimal format.

    Returns:
    - boxes: The list of bounding boxes for each detected face.
    - image_with_boxes: The image with the bounding boxes drawn and the strategies applied.
    """

    # Make prediction
    _, boxes, image_with_boxes = predict(
        frame,
        model,
        names,
        faces,
        scale_coords,
        non_max_suppression,
        plot_one_box,
        strategies,
        background,
    )

    # Return the list of bounding boxes and the image with the bounding boxes drawn and the strategies applied
    return boxes, image_with_boxes


class VideoTransformer(VideoTransformerBase):
    """
    A class used to transform video frames for object detection.

    ...

    Attributes
    ----------
    model : object
        The object detection model to use.
    names : list
        The names of the classes that the model can predict.
    faces : list
        A list to store the detected faces.
    scale_coords : function
        The function to rescale the coordinates to the original image size.
    non_max_suppression : function
        The non-maximum suppression function to apply to the model's predictions.
    plot_one_box : function
        The function to draw a bounding box on the image.
    strategies : list
        A list of strategies to apply to the image.
    background : str
        The desired background color in hexadecimal format.

    Methods
    -------
    transform(frame)
        Processes a frame for object detection and applies various strategies.
    """

    def __init__(
        self,
        model,
        names,
        faces,
        scale_coords,
        non_max_suppression,
        plot_one_box,
        strategies=None,
        background="#56ecd5",
    ):
        self.model = model
        self.names = names
        self.faces = faces
        self.non_max_suppression = non_max_suppression
        self.scale_coords = scale_coords
        self.plot_one_box = plot_one_box
        self.strategies = strategies if strategies is not None else []
        self.background = background

    def transform(self, frame):
        """
        Processes a frame for object detection and applies various strategies.

        Parameters:
        - frame: The original frame.

        Returns:
        - boxes: The list of bounding boxes for each detected face.
        - img: The image with the bounding boxes drawn and the strategies applied.
        """

        img = frame.to_ndarray(format="bgr24")

        # Process the image using the process_frame function
        boxes, img = process_frame(
            img,
            self.model,
            self.names,
            self.faces,
            self.non_max_suppression,
            self.scale_coords,
            self.plot_one_box,
            self.strategies,
            self.background,
        )
        # for strategy in self.strategies:
        #     if strategy == "blur_faces":
        #         img = blur_faces(img, boxes)
        #         cv2.imwrite("output.jpg", img)
        #     if strategy == "whiten_background":
        #         img = whiten_background(img, boxes)

        # Return the list of bounding boxes and the processed image
        return img


def create_videotransformer(
    model,
    names,
    images,
    scale_coords,
    non_max_suppression,
    plot_one_box,
    strategies,
    background="#56ecd5",
):
    """
    Creates a function that returns a VideoTransformer instance.

    Parameters:
    - model: The object detection model to use.
    - names: The names of the classes that the model can predict.
    - images: A list to store the processed images.
    - scale_coords: The function to rescale the coordinates to the original image size.
    - non_max_suppression: The non-maximum suppression function to apply to the model's predictions.
    - plot_one_box: The function to draw a bounding box on the image.
    - strategies: A list of strategies to apply to the image.
    - background: The desired background color in hexadecimal format.

    Returns:
    - _create_videotransformer: A function that returns a VideoTransformer instance when called.
    """

    def _create_videotransformer():
        """
        Returns a VideoTransformer instance with the provided parameters.

        Returns:
        - A VideoTransformer instance.
        """

        # Create a VideoTransformer instance with the provided parameters
        return VideoTransformer(
            model,
            names,
            images,
            scale_coords,
            non_max_suppression,
            plot_one_box,
            strategies,
            background,
        )

    # Return the function that creates a VideoTransformer instance
    return _create_videotransformer
