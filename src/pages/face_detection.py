import streamlit as st
import streamlit.components.v1 as components
import base64
from PIL import Image, ImageOps
import numpy as np

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.utils.video_helper import (
    create_videotransformer,
)
from streamlit_webrtc import webrtc_streamer
from src.utils.model_loader import load_model
from src.utils.predict import predict


def show():
    st.set_page_config(page_title="Detection Page", layout="wide")
    st.markdown(
        "<style>  ul[data-testid=stSidebarNavItems]  {display: none;} </style>",
        unsafe_allow_html=True,
    )
    st.sidebar.title("Navigation")
    images = []
    page = st.sidebar.radio("Go to", ["Detection Page", "Home"])
    st.sidebar.markdown(
        '<hr style="border: 0.9px solid orange">', unsafe_allow_html=True
    )
    if page == "Home":
        st.title("Home Page")
        st.switch_page("pages/home.py")
    if page == "Detection Page":
        st.title("Detection Page")
        st.sidebar.subheader("Detection Configuration")
        detection_type = st.sidebar.radio(
            "Choose detection type",
            ["Upload Image", "Real-Time Detection"],
            index=None,
        )
        with st.container(border=True):
            col1, col2 = st.columns(2, gap="large")
            with col1:
                st.title("Original Image")
                if detection_type == None:
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
                (
                    model,
                    names,
                    scale_coords,
                    non_max_suppression,
                    plot_one_box,
                ) = load_model("src/models/model.pt")
                if detection_type == "Upload Image":
                    uploaded_file = st.sidebar.file_uploader(
                        "Choose an image...", type=["jpg", "jpeg", "png"]
                    )
                    img_array = None
                    if uploaded_file is not None:
                        # Convert the file to an image
                        img_str = base64.b64encode(uploaded_file.read()).decode()
                        img = Image.open(uploaded_file)
                        # Convert the image to a numpy array
                        img_array = np.array(img)
                    else:
                        img_str = base64.b64encode(
                            open("static/images/img_uplod.jpg", "rb").read()
                        ).decode()
                    # Create the HTML for the image container
                    img_container = f"""
                        <div style="
                            width: 100%px;
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
                    if st.button("Predict", use_container_width=True):
                        st.write("predict button is pressed")
                        _, _, image_with_boxes = predict(
                            img_array,
                            model,
                            names,
                            images,
                            scale_coords,
                            non_max_suppression,
                            plot_one_box,
                        )
                        # if len(images) > 0:
                        #     images.append(image_with_boxes)
                        st.write(len(images))
                if detection_type == "Real-Time Detection":
                    st.sidebar.write("Real-Time Detection is selected")
                    st.header("Real Time Detection")
                    strategies = st.sidebar.multiselect(
                        "Select the features to apply",
                        options=[
                            "blur_faces",
                            "whiten_background",
                        ],
                        default=["blur_faces"],
                        label_visibility="visible",
                    )
                    webrtc_streamer(
                        key="example",
                        video_transformer_factory=create_videotransformer(
                            model,
                            names,
                            images,
                            scale_coords,
                            non_max_suppression,
                            plot_one_box,
                            strategies,
                        ),
                    )

            with col2:
                st.title("Detected Face")
                st.subheader("Number of detected faces : " + str(len(images)))
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
                    # uploaded_files = st.file_uploader(
                    #     "Choose images for the new container...",
                    #     type=["jpg", "jpeg", "png"],
                    #     accept_multiple_files=True,
                    #   )
                    # Calculate the number of images per row
                    images_per_row = int(np.ceil(np.sqrt(len(images))))
                    if images_per_row > 0:
                        for i in range(len(images)):
                            image = Image.fromarray(images[i])
                            # Resize the image to a fixed size
                            images[i] = ImageOps.fit(image, (100, 100))
                        # Display the images in rows
                        for i in range(0, len(images), images_per_row):
                            cols = st.columns(images_per_row)
                            for j in range(images_per_row):
                                idx = i + j
                                if idx < len(images):
                                    cols[j].image(images[idx], use_column_width=True)


show()
