import streamlit as st
import streamlit.components.v1 as components
import base64
from PIL import Image


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
            with col2:
                st.title("Detected Face")
                # components.html(img_container, height=320)
        if detection_type == "Real-Time Detection":
            st.sidebar.write("Real-Time Detection is selected")

    if page == "About":
        st.title("About Page")

    if page == "Help":
        st.title("Help Page")


show()
