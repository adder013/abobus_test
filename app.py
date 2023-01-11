import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

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