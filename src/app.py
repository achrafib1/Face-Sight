import streamlit as st

# Page title
st.set_page_config(page_title="Face Sight!", layout="centered")
st.markdown(
    """
    <style>
    a[href$="/"] {display: none;}

        a[href*="/home"]{
            display: none;
        }
        a[href*="/face_detection"]{
            display: none;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def show():
    # Page title
    st.markdown(
        "<h1 style='text-align: center; color: black;'>Face Sight</h1>",
        unsafe_allow_html=True,
    )

    st.image("static/images/face_det_image (2).jpg")
    if st.button("Start", use_container_width=True):
        st.switch_page("pages/home.py")


show()
