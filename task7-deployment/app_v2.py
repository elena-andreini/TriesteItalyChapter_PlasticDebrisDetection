"""
Plastic Debris Detection Streamlit Application

This Streamlit application is designed to detect plastic debris in satellite imagery 
of the Italian and Mediterranean Seas using a pre-trained machine learning model.

Key Features:
- Upload Sentinel-2 satellite imagery (.tif/.tiff files)
- Process images through a trained plastic debris detection model
- Visualise original imagery with overlaid prediction heatmap
- Adjust prediction opacity for better visualisation

Collaborators: Willow Mahoney, Sarah Heyman, [add additional collaborators here]
Project: Omdena Trieste Italy Chapter - Plastic Debris Detection
Date: April 2025
"""

# 1. IMPORTS
#------------
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import streamlit as st
from model import predict
import folium
from streamlit_folium import st_folium


# 2. PAGE CONFIGURATION
#---------------------
st.set_page_config(
    page_title="Plastic Debris Detector",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# 3. HELPER FUNCTIONS
#--------------------

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


# Add more functions here as needed to visualise data, calculate statistics e.g. estimated debris coverage in square metres


# 4. PAGE TITLE AND INTRODUCTION
#-------------------------------
st.title("Detecting Plastic Debris through Satellite Imagery in the Italian and Mediterranean Seas")
st.markdown("""
This application uses a machine learning model to detect and highlight areas with plastic debris 
in Sentinel-2 satellite imagery. Upload an image or select a region of interest to begin.
""")


# 5. SIDEBAR ELEMENTS
#--------------------

# Input Options - Upload a File or Select Region by Coordinates

st.sidebar.header("Input Options")

input_method = st.sidebar.radio("Select Input Method:", 
                             ["Upload Sentinel-2 Image", "Select Region by Coordinates"])

# Additional options - Modify Detection Threshold

st.sidebar.header("Select Detection Threshold")

detection_threshold = st.sidebar.slider(
    "Detection Confidence Threshold", 
    min_value=0.0, 
    max_value=1.0, 
    value=0.5,
    step=0.05
)

st.sidebar.header("Visualisation Options")

visualisation_style = st.sidebar.selectbox(
    "Visualisation Style",
    ["Heatmap", "Bounding Boxes", "Combined"]
)

prediction_opacity = st.sidebar.slider("Prediction Opacity", 0.0, 1.0, 0.5)

# Information about the model

st.sidebar.header("Model Information")
st.sidebar.info(
    """
    This application uses a deep learning model trained to detect plastic debris in 
    Sentinel-2 satellite imagery. The model was trained on [to be added]
    
    Note: This is a demonstration version. In the production environment, the model would 
    be integrated with Sentinel-2 data APIs.
    """
)

# Add model details

with st.sidebar.expander("Model Details"):
    st.write("""
    - **Architecture**: [to be added]
    - **Input**: Sentinel-2 multispectral imagery (bands: B2, B3, B4 [add more here]) 
    - **Output**: Pixel-wise probability of plastic debris
    - **Resolution**: [to be added] m per pixel
    """)


# 6. DATA LOADING AND PROCESSING
#-------------------------------

# Load model

# ADD CODE HERE


# Main area for file upload and image display
if input_method == "Upload Sentinel-2 Image":

    st.subheader("Upload a Sentinel-2 Image")
    uploaded_file = st.file_uploader("Choose a satellite image file", type=["tif", "tiff"])
    if uploaded_file:
        with rasterio.open(uploaded_file) as file:
            data = file.read()

        image_data = data_to_image(data)
        #show_image(image_data)

    with st.spinner("Detecting plastic...", show_time=True):
        predictions = predict(data)

    show_image(predictions * prediction_opacity + image_data * (1.0 - prediction_opacity))

    # Add statistics here

elif input_method == "Select Region by Coordinates":
    st.subheader("Select Region by Coordinates")
    
    # Get coordinates from user
    # Currently via lat/lon but could try changing so user selects region with cursor on map
    col1, col2 = st.columns(2)
    with col1:
        lat = st.number_input("Latitude", value=38.0, format="%.6f")
        lon = st.number_input("Longitude", value=8.0, format="%.6f")
    
    with col2:
        size = st.slider("Area Size (km)", 1, 10, 5)

    # Create a map centered at the coordinates
    m = folium.Map(location=[lat, lon], zoom_start=6)
    folium.Marker([lat, lon], popup="Selected Location").add_to(m)  

    # Add a rectangle to show the area of interest
    folium.Rectangle(
        #Below, bounds uses an approximation for quick calculation
#By dividing the size (in km) by 111, we are converting kilometers to decimal degrees
        bounds=[[lat - size/111, lon - size/111], [lat + size/111, lon + size/111]],
        fill=True,
        color='red',
        fill_opacity=0.2
    ).add_to(m)
    
    # Display the map
    st.subheader("Selected Region")
    st_folium(m, width=700)
    
    if st.button("Fetch Sentinel-2 Data and Process"):
        with st.spinner("Fetching satellite data and processing..."):

            # This would connect to a Sentinel-2 API if setup oeprationally
            st.info("In the production app, this would fetch actual Sentinel-2 data from APIs")

        # Add code here to display satellite image, run detection, overlay heatmap with prediction opacity and calculate statistics


# Footer
st.markdown("---")
