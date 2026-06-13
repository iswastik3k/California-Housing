import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from pathlib import Path
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from src.utils.config import load_config


def plot_residuals(y_true, y_pred, save_path=None):
    residuals = y_true - y_pred
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_pred, y=residuals, alpha=0.6)
    plt.axhline(0, color="red", linestyle="--")
    plt.title("Residuals vs Predictions")
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_feature_importance(model, X, save_path=None):
    importance = model.feature_importances_
    features = X.columns
    fi_df = pd.DataFrame({"Feature": features, "Importance": importance})
    fi_df = fi_df.sort_values("Importance", ascending=False)

    plt.figure(figsize=(8, 6))
    sns.barplot(x="Importance", y="Feature", data=fi_df, color="skyblue")
    plt.title("Feature Importance (XGBoost)")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_shap(model, X, reports_dir):
    # Cast to float to avoid dtype errors
    X_numeric = X.astype(float)

    # Use TreeExplainer directly for XGBoost
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_numeric)

    # Bar plot
    shap.plots.bar(shap_values, show=False)
    plt.tight_layout()
    plt.savefig(reports_dir / "shap_importance.png")
    plt.show()

    # Summary beeswarm plot
    shap.summary_plot(shap_values, X_numeric, show=False)
    plt.tight_layout()
    plt.savefig(reports_dir / "shap_summary.png")
    plt.show()


def evaluate_model(config_path="configs/data_config.yaml", model_path="models/xgb_tuned.pkl"):
    # Load config and dataset
    config = load_config(config_path)
    processed_path = config["processed_data_path"]

    df = pd.read_csv(processed_path)
    X = df.drop("median_house_value", axis=1)
    y = df["median_house_value"]

    # Load model
    model = joblib.load(model_path)
    y_pred = model.predict(X)

    # Metrics
    rmse = root_mean_squared_error(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    print(f"Evaluation -> RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.2f}")

    # Ensure reports directory exists
    reports_dir = Path(__file__).resolve().parent.parent.parent / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Residuals plot
    plot_residuals(y, y_pred, save_path=reports_dir / "residuals.png")

    # Feature importance plot
    plot_feature_importance(model, X, save_path=reports_dir / "feature_importance.png")

    # SHAP plots
    plot_shap(model, X, reports_dir)


if __name__ == "__main__":
    evaluate_model()
