import os
import sys
import torch
import random
import shutil
import pandas
import pathlib
import torchvision
from pathlib import Path
import torchvision.datasets as datasets

class DownloadData:
    def __init__(self):
        self.data_dir = pathlib.Path("C:/datascienceprojects/food_image_classification/food_data")

    def download_data(self):
        # Check if the directory exists and has content 
        if not os.path.exists(self.data_dir) or os.stat(self.data_dir).st_size == 0:
            # Download and apply transforms since the directory is empty or doesn't exist
            train_data = datasets.Food101(root=self.data_dir, 
                                            split="train",
                                            download=True)

            test_data = datasets.Food101(root=self.data_dir, 
                                            split="test",
                                            download=True)
            return train_data, test_data
        else:
            pass

class FoodDataOrganizer:
    def __init__(self, data_dir, target_classes, seed=42):
        self.data_dir = pathlib.Path(data_dir)
        self.image_path = self.data_dir / "food-101" / "images"
        self.target_classes = target_classes
        self.seed = seed

    def _get_image_labels(self, data_split):
        label_path = self.data_dir / "food-101" / "meta" / f"{data_split}.txt"
        with open(label_path, "r") as f:
            labels = [line.strip("\n") for line in f.readlines() if line.split("/")[0] in self.target_classes]
        return labels

    def get_random_subset(self, data_splits=["train", "test"], amount=1.0):
        random.seed(self.seed)
        label_splits = {}

        for data_split in data_splits:
            print(f"[INFO] Creating image split for: {data_split}...")
            labels = self._get_image_labels(data_split)

            number_to_sample = round(amount * len(labels))
            print(f"[INFO] Getting random subset of {number_to_sample} images for {data_split}...")
            sampled_images = random.sample(labels, k=number_to_sample)

            image_paths = [pathlib.Path(str(self.image_path / sample_image) + ".jpg") 
                            for sample_image in sampled_images]
            label_splits[data_split] = image_paths

        return label_splits

    def organize_subset(self, label_splits, target_dir_name):
        target_dir = pathlib.Path(target_dir_name)
        print(f"Creating directory: '{target_dir_name}'")

        target_dir.mkdir(parents=True, exist_ok=True)

        for image_split in label_splits.keys():
            for image_path in label_splits[str(image_split)]:
                dest_dir = target_dir / image_split / image_path.parent.stem / image_path.name
                if not dest_dir.parent.is_dir():
                    dest_dir.parent.mkdir(parents=True, exist_ok=True)
                print(f"[INFO] Copying {image_path} to {dest_dir}...")
                shutil.copy2(image_path, dest_dir)
