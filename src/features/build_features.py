import os
import logging
import yaml
import pandas as pd
import numpy as np

logging.basicConfig(
    filename="logs/feature_engineering.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def load_config(config_path="configs/data_config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
    

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply feature engineering directly to a DataFrame."""
    df["rooms_per_household"] = df["total_rooms"] / df["households"]
    df["bedrooms_ratio"] = df["total_bedrooms"] / df["total_rooms"]
    df["population_per_household"] = df["population"] / df["households"]
    df["income_per_capita"] = df["median_income"] / df["households"]
    df["log_population"] = np.log1p(df["population"])
    return df
    
def build_features(config_path="configs/data_config.yaml"):
    """Pipeline runner: load raw CSV, apply features, save processed CSV."""
    config = load_config(config_path)
    raw_path = config["raw_data_path"]
    processed_path = config["processed_data_path"]

    logging.info("Starting feature engineering...")
    df = pd.read_csv(raw_path)

    # Handle missing values in total_bedrooms
    median_bedrooms = df["total_bedrooms"].median()
    df["total_bedrooms"].fillna(median_bedrooms, inplace=True)


    # Derived features
    df["rooms_per_household"] = df["total_rooms"] / df["households"]
    df["bedroom_ratio"] = df["total_bedrooms"] / df["total_rooms"]
    df["population_per_household"] = df["population"] / df["households"]
    df["income_per_capita"] = df["median_income"] / df["households"]
    df["log_population"] = np.log1p(df["population"])

    # One hot encode categorical
    df = pd.get_dummies(df, columns=["ocean_proximity"], drop_first=True)

    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    df.to_csv(processed_path, index=False)

    logging.info(f"Processed dataset saved to {processed_path}")
    print(f"Processed dataset saved to {processed_path}")

if __name__ == "__main__":
    build_features()