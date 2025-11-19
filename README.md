# **NeuroForecast 🧠📈**

**NeuroForecast** is a lightweight "Mini AutoML" library for time series forecasting. It leverages Large Language Models (LLMs) to act as the data scientist—analyzing your dataset's statistical profile and recommending the best algorithm (LightGBM, XGBoost, Prophet, or SVM) along with optimal hyperparameters and feature engineering strategies.

## **🚀 Features**

* **LLM-Driven Decision Making:** Instead of brute-force grid search, NeuroForecast uses an LLM to analyze data statistics (trend, seasonality, volatility) and pick the best model.  
* **Automated Profiling:** Automatically detects trend direction, seasonality patterns (weekly, yearly), and data quality issues.  
* **Robust Feature Engineering:** Auto-generates:  
  * Lag features (t-1, t-7, t-14...)  
  * Rolling window statistics (mean, std dev)  
  * Date-part features (day of week, quarter, is\_weekend)  
* **Unified API:** Switch between LightGBM, XGBoost, Prophet, and SVM with a single interface.  
* **Dependency Agnostic:** Design your own LLM connector (OpenAI, Anthropic, Gemini, Local Llama) and pass it in as a function.

## **📦 Installation**

NeuroForecast requires **Python 3.8+**.

1. **Clone or copy** neuro\_forecast.py into your project.  
2. **Install dependencies**:

pip install pandas numpy scipy

3. **Optional (Install what you need):**  
   * For Tree models: pip install lightgbm xgboost  
   * For Prophet: pip install prophet  
   * For SVM: pip install scikit-learn

## **⚡ Quick Start**

import pandas as pd  
from neuro\_forecast import NeuroForecaster

\# 1\. Load your data  
df \= pd.read\_csv('sales\_data.csv') \# Must have a date column and a target column

\# 2\. Define your LLM Connector  
\# This function receives a text prompt and must return a JSON string.  
def my\_llm\_connector(prompt):  
    import openai  
    client \= openai.Client(api\_key="...")  
    response \= client.chat.completions.create(  
        model="gpt-4",  
        messages=\[{"role": "user", "content": prompt}\]  
    )  
    return response.choices\[0\].message.content

\# 3\. Initialize & Train  
nf \= NeuroForecaster(llm\_function=my\_llm\_connector)

\# Fits the model (Profiles data \-\> Asks LLM \-\> Generates Features \-\> Trains)  
nf.fit(df, date\_col='date', y\_col='sales')

\# 4\. Predict  
\# Future dataframe must have dates and any external regressors used  
future\_df \= pd.DataFrame({'date': pd.date\_range('2024-01-01', periods=30)})  
forecast \= nf.predict(future\_df)

print(forecast)

## **🛠 Architecture**

### **1\. DataProfiler**

Analyzes the raw time series before any training occurs. It calculates:

* **Trend:** Slope detection.  
* **Seasonality:** Autocorrelation checks at lag 7 (weekly) and 365 (yearly).  
* **Volatility:** Coefficient of variation.  
* **Sparsity:** Missing values and zero counts.

### **2\. The LLM "Brain"**

The profiler sends a summary to the LLM (via your callback function). The LLM returns a JSON plan:

{  
    "model": "lightgbm",  
    "params": { "learning\_rate": 0.05, "n\_estimators": 500 },  
    "feature\_strategy": { "lags": true, "rolling": true, "date\_parts": true },  
    "reasoning": "Data has complex seasonality..."  
}

### **3\. FeatureGenerator**

Based on the LLM's strategy, this component transforms the data:

* **Lags:** Shifts the target variable (e.g., what was sales 7 days ago?).  
* **Rolling:** Calculates moving averages/std devs to capture recent trends.  
* **Date Parts:** Extracts semantic meaning from timestamps (Hour, Day of Week).

### **4\. ModelFactory**

A wrapper layer that standardizes the inputs/outputs for:

* LightGBMWrapper  
* XGBoostWrapper  
* ProphetWrapper  
* SVMWrapper (SVR)

## **📝 Advanced Usage**

### **Using Regressors (External Variables)**

If your dataset has external drivers (e.g., temperature, promotion\_active), pass them during fit:

nf.fit(  
    df,   
    date\_col='date',   
    y\_col='sales',   
    regressors=\['temperature', 'promotion\_active'\]  
)

*Note: These columns must also exist in the future\_df passed to .predict().*

### **Customizing the Prompt**

The library generates the prompt automatically, but the intelligence relies on your llm\_function. You can inject system instructions or debugging prints inside your callback wrapper to control the LLM's behavior.

## **⚠️ Limitations**

* **Recursive Forecasting:** The current implementation uses a direct forecast approach for features. For long horizons with Lag features, values are not recursively updated in the prediction loop (simplification).  
* **Prophet Interface:** Prophet requires specific column names (ds, y). The wrapper handles renaming internally, but ensure your input dates are clean.

## **📄 License**

MIT