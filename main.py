from neuro_forecast import NeuroForecaster
import pandas as pd


# Your LLM API Wrapper
def my_gpt4_call(prompt):
    # call openai api...
    return response_text


# Load Data
df = pd.read_csv("test_data.csv")

# Run
auto_model = NeuroForecaster(llm_function=my_gpt4_call)
auto_model.fit(df, date_col="date", y_col="revenue")

# Create future dataframe
future_df = pd.DataFrame({"date": future_dates})


future_df["marketing_spend"] = [200, 200, 250, 300, 300, 500, 200]  # Planned spend
future_df["local_event"] = [0, 0, 0, 0, 0, 1, 0]  # Known future events


# Forecast
forecast = auto_model.predict(future_df)
