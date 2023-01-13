import streamlit as st
from PIL import Image
import numpy as np
from model import Prediction, MODEL

file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if file is not None:
    image = Image.open(file)

    st.image(
        image,
        caption=f"You amazing image has shape",
        use_column_width=True,
    )

    img_array = np.array(image)
    img = tf.image.resize(img_array, size=(224,224))
    img = tf.expand_dims(img, axis=0)

if st.session_state.question and st.session_state.context:
    answer_dict = (Prediction.get_model()(image=st.session_state.imag))
    st.text_input('Ответ', value=answer_dict['answer'], disabled=True)