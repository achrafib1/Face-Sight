from streamlit_webrtc import VideoTransformerBase
import av
from .image_processor import process_image
from .box_drawer import draw_boxes
from .predict import predict


def process_frame(
    frame, model, names, faces, non_max_suppression, scale_coords, plot_one_box
):
    # Make prediction
    _, image_with_boxes = predict(
        frame, model, names, faces, scale_coords, non_max_suppression, plot_one_box
    )

    return image_with_boxes


class VideoTransformer(VideoTransformerBase):
    def __init__(
        self, model, names, faces, scale_coords, non_max_suppression, plot_one_box
    ):
        self.model = model
        self.names = names
        self.faces = faces
        self.non_max_suppression = non_max_suppression
        self.scale_coords = scale_coords
        self.plot_one_box = plot_one_box

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Process the image
        img = process_frame(
            img,
            self.model,
            self.names,
            self.faces,
            self.non_max_suppression,
            self.scale_coords,
            self.plot_one_box,
        )
        return img
