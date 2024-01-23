import streamlit as st
import streamlit.components.v1 as components


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
        with st.container(border=True):
            img_container = """
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
                <p style="color: grey;">Image will be displayed here</p>
            </div>
            """
            col1, col2 = st.columns(2, gap="large")
            with col1:
                st.title("Original Image")
                components.html(img_container, height=320)
            with col2:
                st.title("Detected Face")
                components.html(img_container, height=320)
        detection_type = st.sidebar.radio(
            "Choose detection type",
            ["Select Option", "Upload Image", "Real-Time Detection"],
        )

        if detection_type == "Upload Image":
            uploaded_file = st.sidebar.file_uploader(
                "Choose an image...", type=["jpg", "jpeg", "png"]
            )

        if detection_type == "Real-Time Detection":
            st.sidebar.write("Real-Time Detection is selected")

    if page == "About":
        st.title("About Page")

    if page == "Help":
        st.title("Help Page")


show()
