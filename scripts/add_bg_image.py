import base64
import streamlit as st

def set_bg_img(image_file) :
    with open(image_file, "rb") as f :
        encoded = base64.b64encode(f.read()).decode()

        css = f"""
        <style>
        .stApp {{
            background-image : url("data:image/png; base64,{encoded}");
            background-size: cover;
            background-repeat: no-repeat;
            background-position: center;
        }}
        </style>
        """
        st.markdown(css ,unsafe_allow_html=True)
    