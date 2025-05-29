import numpy as np


# MARIDA stats
class_distr = np.array([0.00452, 0.00203, 0.00254, 0.00168, 0.00766, 0.15206, 0.20232,
 0.35941, 0.00109, 0.20218, 0.03226, 0.00693, 0.01322, 0.01158, 0.00052])

bands_mean = np.array([0.05197577, 0.04783991, 0.04056812, 0.03163572, 0.02972606, 0.03457443,
 0.03875053, 0.03436435, 0.0392113,  0.02358126, 0.01588816]).astype(np.float32)

bands_std = np.array([0.04725893, 0.04743808, 0.04699043, 0.04967381, 0.04946782, 0.06458357,
 0.07594915, 0.07120246, 0.08251058, 0.05111466, 0.03524419]).astype(np.float32)

MARIDA_LABELS = {
    i: label for i, label in enumerate([
        'Marine Debris', 'Dense Sargassum', 'Sparse Sargassum', 'Natural Organic Material',
        'Ship', 'Clouds', 'Marine Water', 'Sediment-Laden Water', 'Foam', 'Turbid Water',
        'Shallow Water', 'Waves', 'Cloud Shadows', 'Wakes', 'Mixed Water'
    ], 1)
}

# create_LR_dataframe: split_path -> df: [image:mask]

# compute_fdi: tiff_path -> FDI : np.array

# cvt_to_fdi: images -> FDIs: np.array

# compute_ndwi: tiff_path -> ndwi: np.array

# plot_fdi: fdi,ndwi, img_path, mask_path ->

# cvt_rgb: 11bands_img -> rgb_img

# display: images, masks -> plt.plot 

# extract_date_tile: filename -> date, tile

# create_marida_df: data_path, mode -> pd.DataFrame({'image', 'mask', 'conf', 'date', 'tile'})

