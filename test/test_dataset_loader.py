import os
import shutil

import pytest

from rag.constants import DEFAULT_DATASET
from rag.dataset_loader import load_remote_dataset, delete_original_dataset

TEST_DATA_PATH = "test_data"


@pytest.fixture
def mock_dataset_dir():
    os.makedirs(TEST_DATA_PATH, exist_ok=True)
    yield TEST_DATA_PATH
    if os.path.exists(TEST_DATA_PATH):
        shutil.rmtree(TEST_DATA_PATH)


def test_load_remote_dataset():
    dataset = load_remote_dataset(DEFAULT_DATASET)
    assert len(dataset) > 0, "The dataset must not be empty"


def test_delete_original_dataset(mock_dataset_dir):
    file_path = os.path.join(mock_dataset_dir, "test_file.txt")
    with open(file_path, "w") as f:
        f.write("Test content")

    assert os.path.exists(file_path), "The file should exist before being deleted"

    delete_original_dataset(mock_dataset_dir)

    assert not os.path.exists(mock_dataset_dir), "The directory should have been removed"
