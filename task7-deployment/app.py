import matplotlib.pyplot as plt
import numpy as np
import rasterio
import streamlit as st

import time  #TODO: Remove once the model is in place

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


#TODO: Cache result when further along in testing/prototyping
def process_data(data):
    # Pretend to do some computation.
    time.sleep(1)

    # For now, just make a mostly-black result with a small white area
    # as our fake prediction.
    predictions = np.zeros(shape=data.shape[1:])  # All black

    # White area (our positives)
    predictions[
        (data.shape[1] // 3):(2 * data.shape[1] // 3),  # Middle third
        (data.shape[2] // 3):(2 * data.shape[2] // 3)   # Middle third
    ] = 1

    # Repeat single channel to make a black-and-white RGB image.
    predictions = np.stack([predictions for _ in range(3)], axis=-1)

    return predictions


uploaded_file = st.file_uploader("Choose a satellite image file", type=["tif", "tiff"])
if uploaded_file:
    with rasterio.open(uploaded_file) as file:
        data = file.read()

    image_data = data_to_image(data)
    #show_image(image_data)

    with st.spinner("Detecting plastic...", show_time=True):
        predictions = process_data(data)

    prediction_opacity = st.slider("Prediction Opacity", 0.0, 1.0, 0.5)

    show_image(predictions * prediction_opacity + image_data * (1.0 - prediction_opacity))

    

    
    
