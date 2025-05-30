import numpy as np
import rasterio

import torch
from torch.utils.data import Dataset
from torchvision import transforms


class MergedSegmentationDataset(Dataset):
    def __init__(
            self, 
            image_paths,
            dataset_ids,
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
        self._dataset_ids = dataset_ids
        self._image_paths = image_paths
        self._impute = np.tile(self._means[:, np.newaxis, np.newaxis], (1, self._image_size, self._image_size))
        

    def __len__(self):
        return len(self._image_paths)

    @staticmethod
    def get_invalid_mask(image, no_data):
        invalid_mask = image == no_data
        invalid_mask |= np.isnan(image)
        invalid_mask |= image < -1.5
        invalid_mask |= image > 1.5
        return invalid_mask.astype(bool)


    def __getitem__(self, idx):
        image_path = self._image_paths[idx]
        mask_path = image_path.replace(".tif", "_cl.tif")
        with rasterio.open(image_path) as src:
            image = src.read(list(self._bands + 1))
            invalid_mask = self.get_invalid_mask(image, src.nodata)
        with rasterio.open(mask_path) as src:
            mask = src.read().astype(int)

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

        return image, mask, self._dataset_ids[idx]
        return torch.from_numpy(image).float(), torch.from_numpy(mask).long(), self._dataset_ids[idx]