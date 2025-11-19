from neuro_forecast import NeuroForecaster
import pandas as pd


# Your LLM API Wrapper
def my_gpt4_call(prompt):
    # call openai api...
    return response_text


# Load Data
df = pd.read_csv("my_data.csv")

# Run
auto_model = NeuroForecaster(llm_function=my_gpt4_call)
auto_model.fit(df, date_col="date", y_col="revenue")

# Forecast
forecast = auto_model.predict(future_df)
