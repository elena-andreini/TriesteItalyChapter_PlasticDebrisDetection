import numpy as np
import rasterio

import torch
from torch.utils.data import Dataset
from torchvision import transforms

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


# download both datasets
# create a unified splits txt files


class MergedSegmentationDataset(Dataset):
    def __init__(
            self, 
            image_paths,
            band_means, 
            band_stds, 
            selected_bands=list(range(11)),
            transform=None,
            standardization=None,
            image_size=256
        ):

        self._image_size = image_size
        self._bands = selected_bands
        self._means = band_means[selected_bands]
        self._stds = band_stds[selected_bands]
        self._transform = transforms.ToTensor() if transform is None else transform
        self._standardization = standardization
        self._image_paths = image_paths
        self._impute = np.tile(self._means[:, np.newaxis, np.newaxis], (1, self._image_size, self._image_size))
        

    def __len__(self):
        return len(self._image_paths)

    @staticmethod
    def get_invalid_mask(image, no_data):
        invalid_mask = image == no_data
        invalid_mask |= np.isnan(image)
        invalid_mask |= image < -1.5
        invalid_mask |= image > -1.5
        return invalid_mask.astype(bool)


    def __getitem__(self, idx):
        image_path = self._image_paths[idx]
        mask_path = image_path.replace(".tif", "_cl.tif")
        # return image_path, mask_path
        with rasterio.open(image_path) as src:
            image = src.read(list(self._bands + 1))
            invalid_mask = self.get_invalid_mask(image, src.nodata)

        with rasterio.open(mask_path) as src:
            mask = src.read().astype(int)

        print(image.shape, mask.shape, invalid_mask.shape)
        mask[invalid_mask.any(axis=0, keepdims=True)] = 0 # as if unlabeled
        image[invalid_mask] = self._impute[invalid_mask]        

        if self._transform is not None:
            stack = np.concatenate([image, mask], axis=0).astype(np.float32)
            stack = np.transpose(stack, (1, 2, 0))
            stack = self._transform(stack)
            image = stack[:-1, :, :].float()
            mask = stack[-1, :, :].long()
            del stack

        if self._standardization is not None:
            image = self._standardization(image)
        
        return image, mask


    # utils ??

    # create_LR_dataframe: split_path -> df: [image:mask]

    # compute_fdi: tiff_path -> FDI : np.array

    # cvt_to_fdi: images -> FDIs: np.array

    # compute_ndwi: tiff_path -> ndwi: np.array

    # plot_fdi: fdi,ndwi, img_path, mask_path ->

    # cvt_rgb: 11bands_img -> rgb_img

    # display: images, masks -> plt.plot 

    # extract_date_tile: filename -> date, tile

    # create_marida_df: data_path, mode -> pd.DataFrame({'image', 'mask', 'conf', 'date', 'tile'})
