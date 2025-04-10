# Plastic Debris Detection Deployment

This project is part of the Omdena Trieste Italy Chapter initiative to detect plastic debris in the Italian and Mediterranean Seas using Sentinel-2 satellite imagery and a machine learning model.

## Deployment Instructions

### 1. Prerequisites

Ensure you have the following installed on your system:
- Python 3.8 or higher
- pip (Python package manager)

### 2. Clone the Repository
Clone the project repository to your local machine:
```bash
git clone <repository-url>
cd task7-deployment
```
If you want to work with seperate branch, then: 

```bash

git clone -b [branch-name] <repository-url>

```

### 3. Install Dependencies
Install the required Python packages:
```bash

pip install -r requirements.txt

```

### 4. Run the Application
Start the Streamlit application:

```bash 
streamlit run app.py 
# Currently working on couple of other files such as app_v2.py

```

### 5. Access the Application
Once the application is running, open your browser and navigate to:

```bash
http://localhost:8501
```

### 6. Features
Upload Sentinel-2 satellite imagery for plastic debris detection.
Select a region of interest using latitude, longitude, and area size.
Visualize predictions as heatmaps or bounding boxes.
Calculate estimated debris coverage in square meters.

### 7. Troubleshooting
If the application fails to start, ensure all dependencies are installed correctly.
Check for Python version compatibility.
For issues with satellite imagery, ensure the uploaded file is in .tif or .tiff format.

### 8. Acknowledgments
This project is a collaborative effort by the Omdena Trieste Italy Chapter. Special thanks to all contributors and collaborators. 

#### Main Repository :
[Trieste Italy Chapter - Plastic Debris Detection](https://github.com/elena-andreini/TriesteItalyChapter_PlasticDebrisDetection)