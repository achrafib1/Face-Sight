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

    return boxes, image_with_boxes


class VideoTransformer(VideoTransformerBase):
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
        img = frame.to_ndarray(format="bgr24")

        # Process the image
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
    def _create_videotransformer():
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

    return _create_videotransformer
