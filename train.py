from data import *
from    models import *
from evaluation import *

import os
import pandas as pd
import numpy as np

from torch.utils.data import DataLoader


train_paths, train_ids = load_metadata(mode="train")
val_paths, val_ids = load_metadata(mode="val")

print("split files loaded")

selected_bands = np.array([ 4, 6, 8, 11]) - 1 #bands conted from 0

train_transform = transforms.Compose([transforms.ToTensor(),
                                    RandomRotationTransform([-90, 0, 90, 180]),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomVerticalFlip()
                                    ])

test_transform = transforms.Compose([transforms.ToTensor()])

train_dataset = MergedSegmentationDataset(
            train_paths,
            train_ids,
            band_means, 
            band_stds, 
            selected_bands=selected_bands,
            transform=None,
            standardization=None
        )

val_dataset = MergedSegmentationDataset(
            val_paths,
            val_ids,
            band_means, 
            band_stds, 
            selected_bands=selected_bands,
            transform=None,
            standardization=None
        )

standardization = transforms.Normalize(band_means[selected_bands].tolist(), band_stds[selected_bands].tolist())


batch_size = 16
train_loader = DataLoader(train_dataset,
                        batch_size=batch_size, 
                        shuffle=True,
                        collate_fn=collate_fn
                        )


test_loader = DataLoader(val_dataset, 
                        batch_size=batch_size,
                        shuffle=False,
                        collate_fn=collate_fn
                        )

print("data loaders initiated")

import time


its = 12
start = time.perf_counter()
for image, target in train_loader:
    print(time.perf_counter() - start)
    print(its)
    its -= 1
    if not its:
        break
    start = time.perf_counter()