import os
import torch
from PIL import Image
from ultralytics import YOLO
import numpy as np

print("CUDA Available:", torch.cuda.is_available())
print("CUDA Version:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))

test_path = os.path.join(base_dir, 'test')
train_path = os.path.join(base_dir, 'train')

test_cat_path = os.path.join(test_path, 'cats')
test_dog_path = os.path.join(test_path, 'dogs')
train_cat_path = os.path.join(train_path, 'cats')
train_dog_path = os.path.join(train_path, 'dogs')

test_image_files = [f for f in os.listdir(test_cat_path)]
test_image_files.extend([f for f in os.listdir(test_dog_path)])

train_image_files = [f for f in os.listdir(train_cat_path)]
train_image_files.extend([f for f in os.listdir(train_dog_path)])

print('size test: ' + np.size(test_image_files))
print('size train: ' + np.size(train_image_files))

model = YOLO('yolov8s.pt')
for i in range(3):
    results = model(os.path.join(test_cat_path, test_image_files[i]))
    results[0].show()