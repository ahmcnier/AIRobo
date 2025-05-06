import os
import torch

print("CUDA Available:", torch.cuda.is_available())
print("CUDA Version:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))

    test_path = os.path.join(base_dir, 'test')
    train_path = os.path.join(base_dir, 'train')

    test_cat_path = os.path.join(test_path, 'cat')
    test_dog_path = os.path.join(test_path, 'dog')
    train_cat_path = os.path.join(train_path, 'cat')
    train_dog_path = os.path.join(train_dog_path, 'dog')
