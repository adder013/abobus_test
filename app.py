import streamlit as st
from PIL import Image
import numpy as np
from model import Prediction

file = st.file_uploader("Upload an image", key="image",  type=["png", "jpg", "jpeg"])

if file is not None:
    image = Image.open(file)

    st.image(
        image,
        caption=f"You amazing image has shape",
        use_column_width=True,
    )

if st.session_state.image:
    answer_dict = (Prediction.get_prediction(image))
    st.text_input('Ответ', value=answer_dict['answer'], disabled=True)
