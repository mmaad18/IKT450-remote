import os

import torch
from torchvision.transforms import v2
import numpy as np
from PIL import Image


def images_size(root_path: str):
    sizes = []
    paths = []

    for folder in os.listdir(root_path):
        folder_path = os.path.join(root_path, folder)
        for file in os.listdir(folder_path):
            if file.endswith(".png"):
                file_path = os.path.join(folder_path, file)
                with Image.open(file_path) as img:
                    sizes.append(img.size)
                    paths.append(file_path)

    return np.array(sizes), paths


def images_size_by_class(root_path: str):
    sizes = []
    classes = []
    paths = []

    for folder in os.listdir(root_path):
        folder_path = os.path.join(root_path, folder)
        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                if file.endswith(".png"):
                    file_path = os.path.join(folder_path, file)
                    with Image.open(file_path) as img:
                        sizes.append(img.size)
                        classes.append(folder)
                        paths.append(file_path)

    return np.array(sizes), classes, paths


def path_to_fish_id(path: str):
    file_name = path.split('\\')[-1]
    return int(file_name.split('_')[-1].split('.')[0])


def get_transform():
    return v2.Compose([
        v2.ToDtype(torch.float32, scale=True),
        v2.RandomResizedCrop(size=(64, 64)),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ColorJitter(brightness=.25, hue=.15),
        v2.GaussianNoise(mean=0.0, sigma=0.05),
        v2.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 1.)),
    ])


def get_root_path():
    return "/home/ubuntu/Documents/Datasets/Fish_GT/image_cropped"

