import streamlit as st
import time


def show():
    # Page title
    st.set_page_config(page_title="Face Sight !", layout="centered")

    # Hide the app page
    st.markdown(
        """
    <style>
    a[href$="/"] {display: none;}

    </style>
    """,
        unsafe_allow_html=True,
    )
    st.markdown("<style> ul {display: none;} </style>", unsafe_allow_html=True)
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Detection Page"])

    if page == "Home":
        with st.container():
            st.title("Welcome to Face Sight !")

        # Introduction
        with st.container():
            st.header("Introduction", divider="orange")
            page_bg_img = "static/images/face_det_image (1).jpg"
            st.write(
                """
        # Welcome to FaceSight!
        FaceSight is a high-performance face detection application that brings the unseen to light.
        It uses state-of-the-art machine learning algorithms to detect faces in images with remarkable speed and accuracy.
        Whether you're scanning through your photo library or streaming video in real-time, FaceSight is designed to spot every face, every time.
        """
            )
        st.image(page_bg_img)
        # Navigation
        with st.container():
            st.header("start spy")
            if st.button("Start", use_container_width=True):
                st.markdown(
                    f"""<style>
                                [data-testid="stAppViewContainer"] > .main {{
                                background-color: rgba(0,0,0,0.2);
                                }}
                        </style>
                                """,
                    unsafe_allow_html=True,
                )
                with st.spinner("Let's start our spying journey!"):
                    # Simulate a delay
                    time.sleep(2)
                st.markdown(
                    f"""<style>
                                [data-testid="stAppViewContainer"] > .main {{
                                background-color: rgba(0,0,0,0);
                                }}
                        </style>
                                """,
                    unsafe_allow_html=True,
                )
                st.switch_page("pages/face_detection.py")
    if page == "Detection Page":
        st.switch_page("pages/face_detection.py")

    # Footer
    with st.container():
        st.subheader("", divider="orange")
        st.write(
            """
        Made with ❤️ by name
        """
        )


show()
