from streamlit_webrtc import VideoTransformerBase
import av
from .image_processor import process_image
from .box_drawer import draw_boxes
from .predict import predict
import cv2
import numpy as np


def process_frame(
    frame, model, names, faces, non_max_suppression, scale_coords, plot_one_box
):
    # Make prediction
    _, boxes, image_with_boxes = predict(
        frame, model, names, faces, scale_coords, non_max_suppression, plot_one_box
    )

    return boxes, image_with_boxes


def blur_faces(image, boxes):
    for x1, y1, x2, y2 in boxes:
        face = image[y1:y2, x1:x2]
        blurred_face = cv2.GaussianBlur(face, (99, 99), 30)
        image[y1:y2, x1:x2] = blurred_face
    return image


def whiten_background(image, boxes):

    # Create a mask of the same size as the image
    mask = np.zeros_like(image)

    # For each detected face, add a white rectangle to the mask
    for x1, y1, x2, y2 in boxes:
        mask[y1:y2, x1:x2] = 255

    # Create a white image of the same size as the image
    white_image = np.ones_like(image) * 255

    # Use the mask to segment the faces from the image
    faces = cv2.bitwise_and(image, image, mask=mask)

    # Use the inverse of the mask to segment the background from the white image
    mask_inv = cv2.bitwise_not(mask)
    background = cv2.bitwise_and(white_image, white_image, mask=mask_inv)

    # Combine the faces and background to create the new image
    new_image = cv2.add(faces, background)

    return new_image


class VideoTransformer(VideoTransformerBase):
    def __init__(
        self,
        model,
        names,
        faces,
        scale_coords,
        non_max_suppression,
        plot_one_box,
        strategy="blur_faces",
    ):
        self.model = model
        self.names = names
        self.faces = faces
        self.non_max_suppression = non_max_suppression
        self.scale_coords = scale_coords
        self.plot_one_box = plot_one_box
        self.strategy = strategy

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
        )
        if self.strategy == "blur_faces":
            img = blur_faces(img, boxes)
        if self.strategy == "whiten_background":
            img = whiten_background(img, boxes)

        return img


def create_videotransformer(
    model, names, images, scale_coords, non_max_suppression, plot_one_box
):
    def _create_videotransformer():
        return VideoTransformer(
            model,
            names,
            images,
            scale_coords,
            non_max_suppression,
            plot_one_box,
        )

    return _create_videotransformer
