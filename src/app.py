import streamlit as st

# Page title
st.set_page_config(page_title="Digit Delineator!", layout="centered")
st.markdown(
    """
    <style>
    a[href$="/"] {display: none;}

        a[href*="/home"]{
            display: none;
        }
        a[href*="/cast_spell"]{
            display: none;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def show():
    # Page title
    st.markdown(
        "<h1 style='text-align: center; color: black;'>Digit Delineator</h1>",
        unsafe_allow_html=True,
    )

    # st.image("static/images/_1eecbcb2-389a-4f9c-bcc2-48bc8cb79fa5.jpg")
    if st.button("Start", use_container_width=True):
        st.switch_page("pages/home.py")


show()
