import streamlit as st
import streamlit.components.v1 as components
import base64
from PIL import Image, ImageOps
import numpy as np

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.utils.video_helper import create_videotransformer
from streamlit_webrtc import webrtc_streamer
from src.utils.model_loader import load_model
from src.utils.predict import predict
from src.utils.box_drawer import blur_faces
import cv2


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
        # Set a subheader in the sidebar
        st.sidebar.subheader("Detection Configuration")
        # Create a radio button in the sidebar to select the detection type
        # The options are "Upload Image" and "Real-Time Detection"
        detection_type = st.sidebar.radio(
            "Choose detection type",
            ["Upload Image", "Real-Time Detection"],
            index=None,
        )
        # Create an expandable section in the sidebar
        features_expander = st.sidebar.expander("Features")
        # create a multi-select box for the user to select the features to apply ,Within the "Features" section
        strategies = features_expander.multiselect(
            "Select the features to apply",
            options=[
                "blur faces",
                "Change Background",
                "change face color",
                "replace faces",
                "highlight edges",
                "pixelate faces",
            ],
            default=["blur faces"],
            label_visibility="visible",
        )
        # Initialize variables for background_image, face_color, and image_replacement
        background_image = ""
        face_color = ""
        image_replacement = ""
        # If the user selects "Change Background" in the strategies multi-select box
        if "Change Background" in strategies:
            # Create a radio button for the user to select the type of background change
            # The options are "Color" and "Image"
            background_type = features_expander.radio(
                "Choose the type of background change",
                ("Color", "Image"),
            )

            # If the user selects "Color" as the background type
            if background_type == "Color":
                # Show a color picker to choose a background color
                color = features_expander.color_picker("Choose a background color")

                # Use the chosen color as the background
                background_image = color

            # If the user selects "Image" as the background type
            if background_type == "Image":
                # Show a file uploader to upload an image as the background
                file = features_expander.file_uploader(
                    "Upload an image as the background"
                )

                # If the user uploads a file
                if file is not None:
                    # Open the image with PIL and convert it to a numpy array
                    background_image = np.array(Image.open(file))

                    # Convert the image from RGB to BGR (since OpenCV uses BGR)
                    background_image = cv2.cvtColor(background_image, cv2.COLOR_RGB2BGR)

        # If the user selects "change face color"  in the strategies multi-select box
        if "change face color" in strategies:
            # Show a color picker to choose a face color
            fcolor = features_expander.color_picker("Choose a background color")

            # Use the chosen color as the background
            face_color = fcolor

        # If the user selects "replace faces"  in the strategies multi-select box
        if "replace faces" in strategies:

            # Display a file uploader to upload an image as the face image replacement
            file = features_expander.file_uploader(
                "Upload an image as the face image replacement"
            )

            # If the user uploads a file
            if file is not None:
                # Open the image with PIL and convert it to a numpy array
                image_replacement = np.array(Image.open(file))

                # Convert the image from RGB to BGR (since OpenCV uses BGR)
                image_replacement = cv2.cvtColor(image_replacement, cv2.COLOR_RGB2BGR)

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
                # Load the model and related methods
                # The loaded items include the model, names, scale_coords, non_max_suppression, and plot_one_box
                (
                    model,
                    names,
                    scale_coords,
                    non_max_suppression,
                    plot_one_box,
                ) = load_model("src/models/model.pt")
                # if the detection type is "Upload Image"
                if detection_type == "Upload Image":
                    # Display a file uploader in the sidebar for the user to upload an image
                    # The accepted file types are jpg, jpeg, and png
                    uploaded_file = st.sidebar.file_uploader(
                        "Choose an image...", type=["jpg", "jpeg", "png"]
                    )
                    # Initialize img_array as None
                    img_array = None
                    # If the user uploads a file
                    if uploaded_file is not None:
                        # Convert the file to an image
                        # Convert the uploaded file to a base64 string
                        img_str = base64.b64encode(uploaded_file.read()).decode()
                        img = Image.open(uploaded_file)
                        # Convert the image to a numpy array
                        img_array = np.array(img)
                    else:
                        # If the user does not upload a file, use a default image located at "static/images/img_uplod.jpg"
                        # Open the default image, read it as bytes, encode it as a base64 string, and decode it to utf-8
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
                    # if the "Predict" button is pressed
                    if st.button("Predict", use_container_width=True):
                        # If the button is pressed, write a message saying "predict button is pressed"
                        st.write("predict button is pressed")
                        st.write(background_image)
                        # Call the predict function with the necessary parameters
                        # The parameters include the image array, the model, the names, the images, the scale_coords, the non_max_suppression, the plot_one_box, the strategies, the background_image, the face_color, and the image_replacement
                        # The function returns three values: v, bx, and image_with_boxes
                        v, bx, image_with_boxes = predict(
                            img_array,
                            model,
                            names,
                            images,
                            scale_coords,
                            non_max_suppression,
                            plot_one_box,
                            strategies,
                            background_image,
                            face_color,
                            image_replacement,
                        )
                        # if len(images) > 0:
                        #     images.append(image_with_boxes)
                        st.write(len(images))
                        # st.write(type(bx))
                        # st.write(bx)
                        # st.image(image_with_boxes)
                # if the detection type is "Real-Time Detection
                if detection_type == "Real-Time Detection":
                    # If "Real-Time Detection" is selected, write a message in the sidebar saying "Real-Time Detection is selected"
                    st.sidebar.write("Real-Time Detection is selected")
                    # Set the header of the main page to "Real Time Detection"
                    st.header("Real Time Detection")
                    # Start a WebRTC streamer with the key "example"
                    # The video_transformer_factory is set to the result of the create_videotransformer function
                    # The create_videotransformer function is called with the necessary parameters, including the model, the names, the images, the scale_coords, the non_max_suppression, the plot_one_box, the strategies, and the background_image
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
                            background_image,
                        ),
                    )

            with col2:
                # Set the title of the column to "Detected Face"
                st.title("Detected Face")
                # Set a subheader with the number of detected faces
                st.subheader("Number of detected faces : " + str(len(images)))
                with st.container(border=True):
                    # Set the background color of the markdown element to light gray
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

                    # If there is at least one image per row
                    if images_per_row > 0:
                        # Loop through each image
                        for i in range(len(images)):
                            # Convert the image from a numpy array to a PIL Image
                            image = Image.fromarray(images[i])
                            # Resize the image to a fixed size
                            images[i] = ImageOps.fit(image, (100, 100))
                        # Display the images in rows
                        for i in range(0, len(images), images_per_row):
                            # Make a new row with the same number of columns as images per row
                            cols = st.columns(images_per_row)
                            # Loop through each column in the row
                            for j in range(images_per_row):
                                # Calculate the index of the image to display
                                idx = i + j
                                # If the index is less than the total number of images
                                if idx < len(images):
                                    # Display the image in the column
                                    cols[j].image(images[idx], use_column_width=True)


show()
