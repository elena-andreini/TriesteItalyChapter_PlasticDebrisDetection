from data import *
from models import *
from evaluation import *

import time
import os
import pandas as pd
import numpy as np

from torch.utils.data import DataLoader
from torch.utils.benchmark import Timer
from tqdm import tqdm

basepath = os.path.dirname(__file__)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")




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


val_loader = DataLoader(val_dataset, 
                        batch_size=batch_size,
                        shuffle=False,
                        collate_fn=collate_fn
                        )

print("data loaders initiated")

t = Timer(
    stmt="dataset[42]",
    globals={"dataset": train_dataset}
)

print(f"dataset sample loading benchmark: {t.timeit(100)}")

lwc_ratio = 20

lwc_size = 1220
marida_size = lwc_size * .5
marida_debris_pixels_distr = np.sum(marida_classes_distr[[0,1,2,3,8]])

effective_ratio = ((1/lwc_ratio) * lwc_size + marida_debris_pixels_distr * marida_size)/(lwc_size + marida_size)
class_distribution = np.array([1 - effective_ratio, effective_ratio])

print(f'class distribution {class_distribution}')

model = ResAttenUNet(len(selected_bands), 2).to(device)
weight = gen_weights(torch.from_numpy(class_distribution), c = 1.03).to(device)
criterion = torch.nn.CrossEntropyLoss(ignore_index=-1, reduction='mean', weight=weight.to(torch.float32))
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3) 

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max',
    factor=0.8,
    patience=4,
    min_lr=8e-7,
    threshold=1e-2,
    verbose=True
)


best_metric = -float('inf')
best_model_path = os.path.join(basepath, "models", 'best_model.pth')
output_classes = 2
metrics_history = []

epochs = 60
total_steps = 0
total_batch_time = 0

start_time = time.perf_counter()
val_time = 0
total_val_time = 0
total_val = 0


def train_step(model, optimizer, criterion):
    optimizer.zero_grad()
    logits = model(image)
    loss = criterion(logits, target)
    loss.backward()
    optimizer.step()
    return loss
        

@torch.no_grad()
def eval_step(model):
    logits = logits.detach()
    logits = torch.movedim(logits, (0,1,2,3), (0,3,1,2))
    logits = logits.reshape((-1,output_classes))
    target = target.reshape(-1)
    mask = target != -1
    logits = logits[mask]
    target = target[mask]

    probs = F.softmax(logits, dim=1).cpu().numpy()
    target = target.cpu().numpy()
    yPredicted += probs.argmax(1).tolist()
    yTrue += target.tolist()


for epoch in range(1, epochs+1):
    model.train()
    pb = tqdm(train_loader, desc=f"epoch {epoch}/{epochs}: ")
    train_metrics = {}

    epoch_start = time.perf_counter()
    for image, target in pb:
        batch_comp_start = time.perf_counter()
        loss = train_step(model, optimizer, criterion)
        batch_time = time.perf_counter()-batch_comp_start

        pb.set_postfix(loss=loss.item(), batch_time=batch_time)
        total_steps += 1
        total_batch_time += batch_time

        val_time = time.perf_counter()


        if epoch % 10 == 0:
            total_val += 1
            with torch.no_grad():
                logits = logits.detach()
                logits = torch.movedim(logits, (0,1,2,3), (0,3,1,2))
                logits = logits.reshape((-1, output_classes))
                target = target.reshape(-1)

                mask = target != -1
                logits = logits[mask]
                target = target[mask]

                probs = F.softmax(logits, dim=1).argmax(1)
                update_metrics(train_metrics)

    if epoch % 10 == 0:
        print('########### Validation Set Evaluation : #############')
        norm_metrics(train_metrics, len(train_dataset))
        print(train_metrics)
    else:
        print(f"epoch time: {time.perf_counter() - epoch_start}")



    if epoch % 10 == 0:
        model.eval()
        val_metrics = {}
        testLossF = []
        valPrecHistory = []
        iters = len(val_loader)
        with torch.no_grad():
            for i, (image, target) in enumerate(val_loader):
                logits = model(image)
                loss = criterion(logits, target)

                logits = torch.movedim(logits, (0,1,2,3), (0,3,1,2))
                logits = logits.reshape((-1,output_classes))
                target = target.reshape(-1)
                mask = target != -1
                
                logits = logits[mask]
                target = target[mask]
                

                probs = F.softmax(logits, dim=1).argmax(1)
                update_metrics(val_metrics)
            
            print('########### Validation Set Evaluation : #############')
            norm_metrics(val_metrics, len(val_loader))
            print(val_metrics)
            metrics_history.append(val_metrics)
            if val_metrics["plastic_debris"]['iou'] > best_metric:
                best_metric = val_metrics["plastic_debris"]['iou']
                torch.save(model.state_dict(), best_model_path)
                print("#"*40)
                print(f"Saved best model with validation metric: {best_metric}")
                print("#"*40)

    total_val_time += time.perf_counter() - val_time
    scheduler.step(val_metrics["plastic_debris"]['iou'])

end_time = time.perf_counter()