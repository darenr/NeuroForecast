import pandas as pd
import numpy as np
import json
from neuro_forecast import NeuroForecaster


# 1. Define the Mock LLM (Same as before, just for the demo)
def mock_llm_brain(prompt_text):
    print("--- Sending Data Profile to LLM ---")
    # For this demo, we force it to pick LightGBM to show regressor handling
    return json.dumps(
        {
            "model": "lightgbm",
            "params": {"n_estimators": 100, "learning_rate": 0.05, "num_leaves": 31},
            "feature_strategy": {"lags": True, "rolling": True, "date_parts": True},
            "reasoning": "Multivariate data detected. LightGBM handles external regressors well.",
        }
    )


# 2. Load the training data
df = pd.read_csv("retail_sales_multivariate.csv")
print(f"Training data ends on: {df['date'].max()}")

# 3. Initialize and Train
nf = NeuroForecaster(llm_function=mock_llm_brain)
nf.fit(df, date_col="date", y_col="revenue", regressors=["marketing_spend", "local_event"])

# 4. Create the future_df
# We want to forecast the next 7 days
last_date = pd.to_datetime(df["date"].max())
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=7, freq="D")

future_df = pd.DataFrame({"date": future_dates})

# CRITICAL: We must populate the regressors for the future dates.
# This is where you input your future plans or scenarios.
future_df["marketing_spend"] = [200, 200, 250, 300, 300, 500, 200]  # Planned spend
future_df["local_event"] = [0, 0, 0, 0, 0, 1, 0]  # Known future events

print("\n--- Future Data (Inputs) ---")
print(future_df)

# 5. Predict
forecast_values = nf.predict(future_df)

# Combine for viewing
future_df["predicted_revenue"] = forecast_values
print("\n--- Final Forecast ---")
print(future_df[["date", "predicted_revenue"]])
