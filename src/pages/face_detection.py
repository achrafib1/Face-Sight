import streamlit as st
import streamlit.components.v1 as components
import base64
from PIL import Image, ImageOps
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
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


def show():
    st.set_page_config(page_title="Detection Page", layout="wide")
    st.markdown("<style> ul {display: none;} </style>", unsafe_allow_html=True)
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["", "Home", "Detection Page", "About", "Help"])
    st.markdown(
        """
    <style>
        div[role="radiogroup"] > label:first-child {
            display: none;
        }
    </style>
    """,
        unsafe_allow_html=True,
    )
    if page == "Home":
        st.title("Home Page")
        st.switch_page("pages/home.py")
    if page == "Detection Page":
        st.title("Detection Page")
        st.sidebar.subheader("Detection Configuration")
        detection_type = st.sidebar.radio(
            "Choose detection type",
            ["Select Option", "Upload Image", "Real-Time Detection"],
        )
        with st.container(border=True):
            col1, col2 = st.columns(2, gap="large")
            with col1:
                st.title("Original Image")
                if detection_type == "Select Option":
                    choice_container = f"""
                            <div style="
                                width: 390px;
                                height: 300px;
                                border: 1px solid black;
                                border-radius: 15px;
                                box-shadow: 5px 5px 5px grey;
                                display: flex;
                                justify-content: center;
                                align-items: center;
                                background-color: #f0f0f0;
                                ">
                                <p>Choose detection type in the side bar config section</p>
                            </div>
                            """
                    components.html(choice_container, height=320)
                if detection_type == "Upload Image":
                    uploaded_file = st.sidebar.file_uploader(
                        "Choose an image...", type=["jpg", "jpeg", "png"]
                    )
                    if uploaded_file is not None:
                        # Convert the file to an image
                        img_str = base64.b64encode(uploaded_file.read()).decode()
                    else:
                        img_str = base64.b64encode(
                            open("static/images/img_uplod.jpg", "rb").read()
                        ).decode()
                        # Create the HTML for the image container
                    img_container = f"""
                        <div style="
                            width: 390px;
                            height: 300px;
                            border: 1px solid black;
                            border-radius: 15px;
                            box-shadow: 5px 5px 5px grey;
                            display: flex;
                            justify-content: center;
                            align-items: center;
                            background-color: #f0f0f0;
                            ">
                            <img src="data:image/png;base64,{img_str}"
                                alt="Uploaded Image" style="width: 100%; 
                                height: 100%;
                                object-fit: cover; 
                                border-radius: 15px; 
                                box-shadow: 5px 5px 5px grey;">
                        </div>
                        """
                    components.html(img_container, height=320)
                if detection_type == "Real-Time Detection":
                    st.sidebar.write("Real-Time Detection is selected")
                    st.header("Real Time Detection")
                    webrtc_streamer(
                        key="example", video_transformer_factory=VideoTransformer
                    )

            with col2:
                st.title("Detected Face")
                with st.container(border=True):
                    st.markdown(
                        f"""
                                <style>
                                [data-testid="stMarkdown"] {{
                                background-color: #f0f0f0;
                                }}
                                </style>
                                """,
                        unsafe_allow_html=True,
                    )
                    uploaded_files = st.file_uploader(
                        "Choose images for the new container...",
                        type=["jpg", "jpeg", "png"],
                        accept_multiple_files=True,
                    )
                    if uploaded_files is not None and len(uploaded_files) > 0:
                        # Calculate the number of images per row
                        images_per_row = int(np.ceil(np.sqrt(len(uploaded_files))))
                        # Create a list to hold the images
                        images = []
                        for uploaded_file in uploaded_files:
                            if uploaded_file is not None:
                                # Convert the file to an image
                                image = Image.open(uploaded_file)
                            else:
                                image = Image.open("static/images/img_uplod.jpg")
                            # Resize the image to a fixed size
                            image = ImageOps.fit(image, (100, 100))
                            images.append(image)
                        # Display the images in rows
                        for i in range(0, len(images), images_per_row):
                            cols = st.columns(images_per_row)
                            for j in range(images_per_row):
                                idx = i + j
                                if idx < len(images):
                                    cols[j].image(images[idx], use_column_width=True)

    if page == "About":
        st.title("About Page")

    if page == "Help":
        st.title("Help Page")


show()
