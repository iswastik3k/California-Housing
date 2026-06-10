import os
import logging
import yaml
import pandas as pd

logging.basicConfig(
    filename="logs/data_acquisition.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def load_config(config_path="configs/data_config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def load_raw_data(config_path="configs/data_config.yaml"):
    config = load_config(config_path)
    raw_path = config["raw_data_path"]

    logging.info("Loading Kaggle dataset...")
    df = pd.read_csv(raw_path)

    logging.info(f"Dataset loaded with shape {df.shape}")
    print("Dataset loaded successfully!")

    return df

if __name__ == "__main__":
    load_raw_data()
