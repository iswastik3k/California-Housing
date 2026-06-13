import pytest
from src.utils.config import load_config

def test_load_config_valid():
    config = load_config("configs/data_config.yaml")
    assert "raw_data_path" in config
    assert "processed_data_path" in config
