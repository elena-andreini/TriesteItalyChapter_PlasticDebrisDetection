import matplotlib.pyplot as plt
import numpy as np
import rasterio
import streamlit as st
from model import predict

st.title("Detecting Plastic Debris through Satellite Imagery in the Italian and Mediterranean Seas")


def data_to_image(data):
    # Note: Image reading adapted from 
    # task4-baseline-modeling/UNet-patches-selection-experiment.ipynb
    bands_to_display = [4, 3, 2,]
    img = np.stack([data[band] for band in bands_to_display], axis=-1)
    img = (img - img.min()) / (img.max() - img.min())

    return img


def show_image(img):
    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.axis('off')  # Hide axes
    st.pyplot(fig)


uploaded_file = st.file_uploader("Choose a satellite image file", type=["tif", "tiff"])
if uploaded_file:
    with rasterio.open(uploaded_file) as file:
        data = file.read()

    image_data = data_to_image(data)
    #show_image(image_data)

    with st.spinner("Detecting plastic...", show_time=True):
        predictions = predict(data)

    prediction_opacity = st.slider("Prediction Opacity", 0.0, 1.0, 0.5)

    show_image(predictions * prediction_opacity + image_data * (1.0 - prediction_opacity))

    

    
    
