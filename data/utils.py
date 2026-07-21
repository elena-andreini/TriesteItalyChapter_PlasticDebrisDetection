import os
import torch
import random
import numpy as np
import pandas as pd
import torchvision.transforms.functional as vF

basepath = os.path.dirname(__file__)

def load_metadata(mode):
    assert mode in ["train", "val"], f"Invalid split {mode}"
    marida = pd.read_csv(os.path.join(basepath, f"marida_{mode}.csv"))
    lwc = pd.read_csv(os.path.join(basepath, f"lwc_{mode}.csv"))

    paths = np.array(marida.image.tolist() + lwc.image.tolist(), dtype="str")
    ids = np.zeros_like(paths, dtype=int)
    ids[:len(marida)] = 1
    perm = np.random.permutation(len(paths))

    paths = paths[perm]
    ids = ids[perm]

    return paths, ids


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def collate_fn(batch, device=device):
    images, masks, dataset_ids = zip(*batch)
    images = torch.stack(images)
    masks = torch.stack(masks)


    images = images.to(device)
    masks = masks.to(device)
    dataset_ids = torch.tensor(dataset_ids, dtype=torch.long, device=device)

    lr_masks = process_lwc(images, masks, dataset_ids, r1=4, r2=16, target_ratio=20, device=device)
    marida_masks = process_marida(masks, dataset_ids, device=device)
    masks = lr_masks + marida_masks
    
    return images, masks



class RandomRotationTransform:
    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return vF.rotate(x, angle)

def gen_weights(class_distribution, c = 1.02):
    return 1/torch.log(c + class_distribution)


def torch_dilate(mask, kernel_size, device="cpu"):
    kernel = torch.ones(1, 1, kernel_size, kernel_size, device=device, dtype=torch.float32)
    mask = mask.float().unsqueeze(1)
    dilated = torch.nn.functional.conv2d(mask, kernel, padding=kernel_size // 2) > 0
    return dilated.squeeze(1).bool()

def process_lwc(images, masks, dataset_ids, r1=4, r2=16, target_ratio=20, threshold=None, device='cpu'):    
    bg_masks = torch.zeros_like(masks, dtype=torch.int64, device=device)

    valid_mask = (dataset_ids == 0)
    if not valid_mask.any():
        return bg_masks


    # selecting lwc samples
    selected_masks = masks[valid_mask]
    bg_masks[valid_mask] = selected_masks * 2
    
    # annular ring
    dilated_r1 = torch_dilate(selected_masks, 2 * r1 + 1, device=device)
    dilated_r2 = torch_dilate(selected_masks, 2 * r2 + 1, device=device)
    annular_masks = dilated_r2 & ~dilated_r1
    
    for idx in range(annular_masks.shape[0]):
        valid_coords = torch.where(annular_masks[idx])
        num_debris = torch.sum(selected_masks[idx] > 0).item()
        num_background = min(len(valid_coords[0]), int(num_debris * target_ratio))
        
        if len(valid_coords[0]) > 0:
            sample_indices = torch.randperm(len(valid_coords[0]), device=device)[:num_background]
            bg_masks[valid_mask.nonzero(as_tuple=True)[0][idx],
                     valid_coords[0][sample_indices],
                     valid_coords[1][sample_indices]] = 1
    
    bg_masks[valid_mask] = bg_masks[valid_mask] - 1
    return bg_masks



def process_marida(masks, dataset_ids, device="cpu"):
    marida_masks = torch.zeros_like(masks, dtype=torch.int64, device=device)
    
    marida_mask = (dataset_ids == 1)
    if not marida_mask.any():
        return marida_masks

    
    selected_masks = masks[marida_mask]
    
    # Set classes [1, 2, 3, 4, 9] to 2
    debris_classes = torch.tensor([1, 2, 3, 4, 9], device=device)
    is_debris = torch.isin(selected_masks, debris_classes)
    marida_masks[marida_mask] = torch.where(
        is_debris,
        torch.tensor(2, dtype=torch.int64, device=device),
        selected_masks
    )

    
    marida_masks[marida_mask] = torch.where(
        (marida_masks[marida_mask] != 0) & (marida_masks[marida_mask] != 2),
        torch.tensor(1, dtype=torch.int64, device=device),
        marida_masks[marida_mask]
    )

    marida_masks[marida_mask] = marida_masks[marida_mask] - 1
    return marida_masks