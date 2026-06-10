import os
import logging
import yaml
from sklearn.datasets import fetch_california_housing
import pandas as pd

# Configure logging
logging.basicConfig(
    filename="logs/data_acquisition.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def load_config(config_path="configs/data_config.yaml"):
    """Load YAML config file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
    
def load_raw_data(config_path="configs/data_config.yaml"):
    config = load_config(config_path)
    save_path = config["raw_data_path"]

    logging.info("Starting data acquisition...")
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)

    logging.info(f"Raw dataset saved to {save_path}")
    print(f"Raw dataset saved to {save_path}")

if __name__ == "__main__":
    load_raw_data()