import os
import shutil

from datasets import load_dataset


def load_remote_dataset(dataset_name):
    dataset = load_dataset(dataset_name)
    return dataset['train'] if 'train' in dataset else dataset


def delete_original_dataset(dataset_path):
    if os.path.exists(dataset_path):
        shutil.rmtree(dataset_path)
