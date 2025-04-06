import matplotlib.pyplot as plt
import numpy as np
import rasterio
import streamlit as st

st.title("Detecting Plastic Debris through Satellite Imagery in the Italian and Mediterranean Seas")

def show_image_data(data):
    # Note: Image reading adapted from 
    # task4-baseline-modeling/UNet-patches-selection-experiment.ipynb
    bands_to_display = [4, 3, 2,]
    img = np.stack([data[band] for band in bands_to_display], axis=-1)
    img = (img - img.min()) / (img.max() - img.min())

    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.axis('off')  # Hide axes
    st.pyplot(fig)

uploaded_file = st.file_uploader("Choose a satellite image file", type=["tif", "tiff"])
if uploaded_file:
    with rasterio.open(uploaded_file) as file:
        data = file.read()
    
    show_image_data(data)
    

    
