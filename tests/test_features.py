import pandas as pd
from src.features.build_features import add_features

def test_add_features_creates_columns():
    df = pd.DataFrame({
        "population": [1000, 2000],
        "households": [100, 200],
        "total_rooms": [500, 1000],
        "total_bedrooms": [200, 400],
        "median_income": [3.5, 4.2]
    })

    df_out = add_features(df)

    # Check engineered columns exist
    assert "rooms_per_household" in df_out.columns
    assert "bedrooms_ratio" in df_out.columns
    assert "population_per_household" in df_out.columns
    assert "income_per_capita" in df_out.columns
    assert "log_population" in df_out.columns

    # Check values are not null
    for col in ["rooms_per_household", "bedrooms_ratio", "population_per_household", "income_per_capita", "log_population"]:
        assert df_out[col].notnull().all()
