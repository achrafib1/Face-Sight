import streamlit as st


def show():
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


show()
