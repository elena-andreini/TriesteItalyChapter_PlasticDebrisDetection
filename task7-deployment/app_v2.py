"""
Plastic Debris Detection Streamlit Application

This Streamlit application is designed to detect plastic debris in satellite imagery
of the Italian and Mediterranean Seas using a pre-trained machine learning model.

Key Features:
- Upload Sentinel-2 satellite imagery (.tif/.tiff files)
- Process images through a trained plastic debris detection model
- Visualise original imagery with overlaid prediction heatmap
- Adjust prediction opacity for better visualisation

Collaborators: Willow Mahoney, Sarah Heyman, Rupak R [add additional collaborators here]
Project: Omdena Trieste Italy Chapter - Plastic Debris Detection
Date: April 2025
"""

# 1. IMPORTS
# ------------
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import streamlit as st
from model import predict
import folium
from streamlit_folium import st_folium


# 2. PAGE CONFIGURATION
# ---------------------
st.set_page_config(
    page_title="Plastic Debris Detector",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# 3. HELPER FUNCTIONS
# --------------------


def data_to_image(data):
    """
    Convert satellite data to an RGB image for visualization.
    Assumes data is a NumPy array with bands indexed by integers.
    """
    # Note: Image reading adapted from
    # task4-baseline-modeling/UNet-patches-selection-experiment.ipynb
    bands_to_display = [4, 3, 2]  # RGB bands
    img = np.stack([data[band] for band in bands_to_display], axis=-1)
    img = (img - img.min()) / (img.max() - img.min())  # Normalize to [0, 1]
    return img


def show_image(img):
    """
    Display an image using Matplotlib and Streamlit.
    """
    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.axis("off")  # Hide axes
    st.pyplot(fig)


def calculate_debris_coverage(predictions, pixel_area, threshold=0.5):
    """
    Calculate the estimated debris coverage in square meters.
    :param predictions: The prediction array (probabilities).
    :param pixel_area: Area of a single pixel in square meters.
    :param threshold: Confidence threshold for debris detection.
    :return: Estimated debris coverage in square meters.
    """
    debris_pixels = np.sum(predictions > threshold)  # Count pixels above the threshold
    debris_coverage = debris_pixels * pixel_area  # Calculate total debris area
    return debris_coverage


# Add more functions here as needed to visualise data, calculate statistics e.g. estimated debris coverage in square metres


# 4. PAGE TITLE AND INTRODUCTION
# -------------------------------
st.title(
    "Detecting Plastic Debris through Satellite Imagery in the Italian and Mediterranean Seas"
)
st.markdown(
    """
This application uses a machine learning model to detect and highlight areas with plastic debris 
in Sentinel-2 satellite imagery. Upload an image or select a region of interest to begin.
"""
)


# 5. SIDEBAR ELEMENTS
# --------------------

# Input Options - Upload a File or Select Region by Coordinates
st.sidebar.header("Input Options")

input_method = st.sidebar.radio(
    "Select Input Method:", ["Upload Sentinel-2 Image", "Select Region by Coordinates"]
)

# Additional options - Modify Detection Threshold
st.sidebar.header("Select Detection Threshold")

detection_threshold = st.sidebar.slider(
    "Detection Confidence Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.05
)

st.sidebar.header("Visualisation Options")

visualisation_style = st.sidebar.selectbox(
    "Visualisation Style", ["Heatmap", "Bounding Boxes", "Combined"]
)

prediction_opacity = st.sidebar.slider("Prediction Opacity", 0.0, 1.0, 0.5)

# Information about the model
st.sidebar.header("Model Information")
st.sidebar.info(
    """
    This application uses a deep learning model trained to detect plastic debris in 
    Sentinel-2 satellite imagery. The model was trained on a dataset of annotated Sentinel-2 images.
    
    Note: This is a demonstration version. In the production environment, the model would 
    be integrated with Sentinel-2 data APIs.
    """
)

# Add model details
with st.sidebar.expander("Model Details"):
    st.write(
        """
    - **Architecture**: U-Net
    - **Input**: Sentinel-2 multispectral imagery (bands: B2, B3, B4, B8) 
    - **Output**: Pixel-wise probability of plastic debris
    - **Resolution**: [to be added] m per pixel
        """
    )


# 6. DATA LOADING AND PROCESSING
# -------------------------------

# Load model
# ADD CODE HERE


# Main area for file upload and image display
if input_method == "Upload Sentinel-2 Image":

    st.subheader("Upload a Sentinel-2 Image")
    uploaded_file = st.file_uploader(
        "Choose a satellite image file", type=["tif", "tiff"]
    )
    if uploaded_file:
        with rasterio.open(uploaded_file) as file:
            data = file.read()  # Read all bands into a NumPy array

        # Convert data to an RGB image
        image_data = data_to_image(data)
        show_image(image_data)

        # Run the prediction model
        with st.spinner("Detecting plastic..."):
            predictions = predict(data)

        # Display results
        st.subheader("Detection Results")
        show_image(
            predictions * prediction_opacity + image_data * (1.0 - prediction_opacity)
        )

        # Calculate pixel area (assuming 10m resolution for Sentinel-2)
        pixel_area = 10 * 10  # 10m x 10m = 100 square meters per pixel
        debris_coverage = calculate_debris_coverage(
            predictions, pixel_area, threshold=detection_threshold
        )
        st.write(f"Estimated debris coverage: {debris_coverage:.2f} square meters")

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
        # Below, bounds uses an approximation for quick calculation
        # By dividing the size (in km) by 111, we are converting kilometers to decimal degrees
        bounds=[
            [lat - size / 111, lon - size / 111],
            [lat + size / 111, lon + size / 111],
        ],
        fill=True,
        color="red",
        fill_opacity=0.2,
    ).add_to(m)

    # Display map
    st.subheader("Selected Region")
    st_folium(m, width=700)

    if st.button("Fetch Sentinel-2 Data and Process"):
        with st.spinner("Fetching satellite data and processing..."):
            st.info(
                "In the production app, this would fetch actual Sentinel-2 data from APIs"
            )

            # Simulate fetching and processing data
            data = np.random.rand(256, 256)  # Simulated satellite data
            predictions = predict(data)

            # Calculate pixel area (assuming 10m resolution for Sentinel-2)
            pixel_area = 10 * 10  # 10m x 10m = 100 square meters per pixel
            # Can be customized later
            debris_coverage = calculate_debris_coverage(
                predictions, pixel_area, threshold=detection_threshold
            )

            # Results
            st.subheader("Detection Results")
            st.write(f"Estimated debris coverage: {debris_coverage:.2f} square meters")

            heatmap_overlay = predictions * prediction_opacity + data * (
                1.0 - prediction_opacity
            )
            show_image(heatmap_overlay)


# Footer
st.markdown("---")
