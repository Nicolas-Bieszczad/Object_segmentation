# -*- coding: utf-8 -*-
"""vnexia_phase3_yolo.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/gist/Nicolas-Bieszczad/dc35c6ee1d3b28708f2d75b5c9e8cdf8/vnexia_phase3_yolo.ipynb

## Drive connection
"""

from google.colab import drive
drive.mount('/content/drive')

"""## Installations and Imports"""

!pip install roboflow
!pip install -U ultralytics wandb

# Enable W&B logging for Ultralytics
!yolo settings wandb=True

!wandb login

import shutil
from roboflow import Roboflow
import os
from ultralytics import YOLO
import matplotlib.pyplot as plt
import shutil
import random
from pathlib import Path
import yaml
import wandb
import torch
import pandas as pd
from wandb.integration.ultralytics import add_wandb_callback

"""## Dataset"""

rf = Roboflow(api_key="api_key")
project = rf.workspace("thesis-2zk7q").project("hq-hole-detection-moving-objects")
version = project.version(9)
dataset = version.download("yolov11")

"""### Modifying dataset split ratio

"""

# Set seed for reproducibility
random.seed(42)

#original dataset location
original_dataset_path = dataset.location

# Output directory
resplit_dataset_path = os.path.join("/content/drive/MyDrive/roboflow_datasets", "resplit")
os.makedirs(resplit_dataset_path, exist_ok=True)

#new folders for train/val/test
splits = ["train", "valid", "test"]
for split in splits:
    os.makedirs(os.path.join(resplit_dataset_path, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(resplit_dataset_path, split, "labels"), exist_ok=True)

# Gather all images and labels from original splits
def collect_images_labels(folder):
    image_dir = os.path.join(original_dataset_path, folder, "images")
    label_dir = os.path.join(original_dataset_path, folder, "labels")
    image_files = sorted([f for f in Path(image_dir).glob("*.jpg")])
    paired = [(img, Path(label_dir) / (img.stem + ".txt")) for img in image_files]
    return paired

all_data = []
for split in ["train", "valid", "test"]:
    all_data.extend(collect_images_labels(split))

print(f"Total examples found: {len(all_data)}")

# Shuffle and split 70/15/15
random.shuffle(all_data)
n_total = len(all_data)
n_train = int(0.70 * n_total)
n_valid = int(0.15 * n_total)
n_test  = n_total - n_train - n_valid

train_data = all_data[:n_train]
valid_data = all_data[n_train:n_train + n_valid]
test_data  = all_data[n_train + n_valid:]

print(f"Split: {len(train_data)} train / {len(valid_data)} valid / {len(test_data)} test")

# Move files to new structure
def move_pairs(pairs, dest_split):
    for img, label in pairs:
        shutil.copy(img, os.path.join(resplit_dataset_path, dest_split, "images", img.name))
        shutil.copy(label, os.path.join(resplit_dataset_path, dest_split, "labels", label.name))

move_pairs(train_data, "train")
move_pairs(valid_data, "valid")
move_pairs(test_data, "test")

# Get class names from original YAML
original_yaml_path = os.path.join(original_dataset_path, "data.yaml")
with open(original_yaml_path, "r") as f:
    original_yaml = yaml.safe_load(f)
class_names = original_yaml["names"]
nc = original_yaml["nc"]

#new data.yaml
resplit_yaml = {
    "path": resplit_dataset_path,
    "train": "train/images",
    "val": "valid/images",
    "test": "test/images",
    "nc": nc,
    "names": class_names,
}

resplit_yaml_path = os.path.join(resplit_dataset_path, "data.yaml")
with open(resplit_yaml_path, "w") as f:
    yaml.dump(resplit_yaml, f)

print(f" New data.yaml saved at: {resplit_yaml_path}")

"""## Hyperparameter optimization and training"""

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

#random search sweep configuration
sweep_config = {
    "method": "random",
    "metric": {
        "name": "metrics/mAP_0.5",
        "goal": "maximize"
    },
    "parameters": {
        "epochs": {"values": [10]},
        "batch": {"values": [16, 32]},
        "lr0": {"values": [1e-3, 1e-4]},
        "weight_decay": {"values": [1e-4, 1e-3]},
        "optimizer": {"values": ["SGD", "Adam"]}
    }
}

data_yaml = "/content/drive/MyDrive/roboflow_datasets/resplit/data.yaml"

#training function used by the WandB agent
def train():
    run = wandb.init(
        project="yolo-seg-worker-randomsearch",
        job_type='training'
    )
    config = run.config
    run_name = f"e{config.epochs}_b{config.batch}_lr{config.lr0}_wd{config.weight_decay}_{config.optimizer}"
    wandb.run.name = run_name

    # Load the model and attach WandB callback
    model = YOLO("/content/yolo11n-seg.pt")
    add_wandb_callback(model, enable_model_checkpointing=True)

    # Train the model using sweep parameters
    model.train(
        data=data_yaml,
        epochs=config.epochs,
        batch=config.batch,
        lr0=config.lr0,
        weight_decay=config.weight_decay,
        optimizer=config.optimizer,
        project="yolo-seg-worker-randomsearch",
        name=run_name,
        device=0 if torch.cuda.is_available() else "cpu",
        val=True
    )

    run.finish()


#Launch sweep
sweep_id = wandb.sweep(sweep_config, project="yolo-seg-worker-randomsearch")
wandb.agent(sweep_id, function=train, count=6)

"""### Training with best hyperparameters"""


EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 0.0001
OPTIMIZER = 'Adam'
MODEL_NAME = 'yolo11n-seg.pt'
PROJECT_NAME = 'yolov11n-segmentation'
RUN_NAME = f"e{EPOCHS}_b{BATCH_SIZE}_lr{LEARNING_RATE}_wd{WEIGHT_DECAY}_{OPTIMIZER}"


wandb.init(project=PROJECT_NAME, name=RUN_NAME, config={
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "learning_rate": LEARNING_RATE,
    "weight_decay": WEIGHT_DECAY,
    "optimizer": OPTIMIZER,
    "model": MODEL_NAME
})

model = YOLO(MODEL_NAME)


results = model.train(
    data="/content/drive/MyDrive/roboflow_datasets/resplit/data.yaml",
    epochs=EPOCHS,
    batch=BATCH_SIZE,
    imgsz=640,
    optimizer=OPTIMIZER,
    lr0=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    project=PROJECT_NAME,
    name=RUN_NAME,
    save=True,
    val=True,
    verbose=True,
    patience=10
)

wandb.finish()

"""## Evaluation on test set"""

# Load trained model weights
trained_model = YOLO("/content/yolov11n-segmentation/e100_b32_lr0.0001_wd0.0001_Adam/weights/best.pt")

# Run evaluation on the test set
test_results = trained_model.val(
    data="/content/drive/MyDrive/roboflow_datasets/resplit/data.yaml",
    split="test",
    imgsz=640,
    batch=32,
    verbose=True
)

print(test_results)