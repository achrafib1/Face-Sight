from streamlit_webrtc import VideoTransformerBase
import av


def process_frame(frame):
    # Make prediction here
    # result = predict(model, frame)

    # Display the result on the frame
    # frame = display_result(frame, result)

    return frame


class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Process the image
        img = process_frame(img)

        return img
