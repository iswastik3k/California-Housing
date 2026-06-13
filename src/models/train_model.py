from xgboost import XGBRegressor
import joblib
import logging
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from src.utils.config import load_config

logging.basicConfig(
    filename="logs/modeling.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def train_final_model(config_path="configs/data_config.yaml"):
    config = load_config(config_path)
    processed_path = config["processed_data_path"]

    df = pd.read_csv(processed_path)
    X = df.drop("median_house_value", axis=1)
    y = df["median_house_value"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Tuned parameters (from 02_modeling.ipynb)
    model = XGBRegressor(
        n_estimators=300,
        max_depth=7,
        learning_rate=0.1,
        subsample=1.0,
        colsample_bytree=0.8,
        random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    rmse = root_mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    logging.info(f"Tuned XGBoost -> RMSE: {rmse}, MAE: {mae}, R²: {r2}")
    print(f"Tuned XGBoost -> RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.2f}")

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/xgb_tuned.pkl")


if __name__ == "__main__":
    train_final_model()