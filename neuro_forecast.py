import pandas as pd
import numpy as np
import json
import warnings
import re
from typing import List, Dict, Any, Optional, Union, Callable
from dataclasses import dataclass
from scipy import stats
from scipy.signal import periodogram

# Import modeling libraries
# In a real package, these would be optional dependencies
try:
    import lightgbm as lgb
except ImportError:
    lgb = None
try:
    import xgboost as xgb
except ImportError:
    xgb = None
try:
    from prophet import Prophet
except ImportError:
    Prophet = None
try:
    from sklearn.svm import SVR
    from sklearn.preprocessing import StandardScaler, OrdinalEncoder
    from sklearn.pipeline import Pipeline
except ImportError:
    SVR = None

warnings.filterwarnings("ignore")


@dataclass
class ModelSuggestion:
    model_type: str
    params: Dict[str, Any]
    reasoning: str
    feature_strategy: Dict[str, bool]


class DataProfiler:
    """Analyzes the time series to generate a descriptive report for the LLM."""

    @staticmethod
    def analyze(
        df: pd.DataFrame, date_col: str, y_col: str, regressors: List[str] = None
    ) -> Dict:
        df = df.sort_values(date_col)
        y = df[y_col].values
        dates = df[date_col]

        # Basic Stats
        n_samples = len(df)
        missing_y = np.isnan(y).sum()
        zeros_y = (y == 0).sum()

        # Trend Analysis (Linear Slope)
        non_nan_idx = ~np.isnan(y)
        if non_nan_idx.sum() > 1:
            slope, _, _, _, _ = stats.linregress(np.arange(len(y[non_nan_idx])), y[non_nan_idx])
            trend_desc = (
                "positive" if slope > 0.01 else "negative" if slope < -0.01 else "flat/stable"
            )
        else:
            trend_desc = "insufficient data"

        # Seasonality Detection (Simple Autocorrelation check)
        # Check lag 7 (weekly) and lag 12/30 (monthly approx)
        seasonality = []
        if n_samples > 30:
            s = pd.Series(y)
            acf_7 = s.autocorr(lag=7)
            acf_12 = s.autocorr(lag=12)
            acf_365 = s.autocorr(lag=365) if n_samples > 730 else 0

            if abs(acf_7) > 0.3:
                seasonality.append("Weekly")
            if abs(acf_12) > 0.3:
                seasonality.append("Monthly/Yearly approx")
            if abs(acf_365) > 0.3:
                seasonality.append("Yearly")

        seasonality_desc = (
            ", ".join(seasonality) if seasonality else "No strong seasonality detected"
        )

        # Volatility / Noise
        if n_samples > 1:
            cv = np.std(y) / (np.mean(y) + 1e-6)  # Coefficient of variation
            volatility = "High" if cv > 1.0 else "Medium" if cv > 0.3 else "Low"
        else:
            volatility = "Unknown"

        return {
            "n_samples": n_samples,
            "start_date": str(dates.min()),
            "end_date": str(dates.max()),
            "missing_values": int(missing_y),
            "zero_values": int(zeros_y),
            "trend": trend_desc,
            "seasonality": seasonality_desc,
            "volatility": volatility,
            "regressors_available": regressors if regressors else "None",
            "data_sample_head": df[[date_col, y_col]].head(5).to_dict(orient="records"),
            "data_sample_tail": df[[date_col, y_col]].tail(5).to_dict(orient="records"),
        }


class FeatureGenerator:
    """Extensive feature engineering engine."""

    def __init__(self, date_col: str, y_col: str, strategy: Dict[str, bool] = None):
        self.date_col = date_col
        self.y_col = y_col
        # Default strategy if LLM doesn't specify
        self.strategy = strategy or {
            "lags": True,
            "rolling": True,
            "date_parts": True,
            "fourier": False,
        }
        self.scalers = {}

    def transform(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        df_out = df.copy()

        # Ensure datetime
        df_out[self.date_col] = pd.to_datetime(df_out[self.date_col])

        # 1. Date Parts
        if self.strategy.get("date_parts"):
            df_out["hour"] = df_out[self.date_col].dt.hour
            df_out["dayofweek"] = df_out[self.date_col].dt.dayofweek
            df_out["quarter"] = df_out[self.date_col].dt.quarter
            df_out["month"] = df_out[self.date_col].dt.month
            df_out["year"] = df_out[self.date_col].dt.year
            df_out["dayofyear"] = df_out[self.date_col].dt.dayofyear
            df_out["is_weekend"] = (df_out["dayofweek"] >= 5).astype(int)

        # 2. Lags & Rolling (Only applicable if we have history in the row context)
        # Note: For recursive forecasting, this requires careful handling in prediction loop.
        # Here we generate them for training sets (Direct forecasting approach assumption).
        if self.strategy.get("lags"):
            for lag in [1, 7, 14, 28]:
                df_out[f"lag_{lag}"] = df_out[self.y_col].shift(lag)

        if self.strategy.get("rolling"):
            # Rolling means excluding current row to prevent leakage
            for window in [7, 30]:
                df_out[f"rolling_mean_{window}"] = (
                    df_out[self.y_col].shift(1).rolling(window=window).mean()
                )
                df_out[f"rolling_std_{window}"] = (
                    df_out[self.y_col].shift(1).rolling(window=window).std()
                )

        # Fill NaNs generated by lags for tree models
        if is_training:
            df_out = df_out.bfill().ffill()

        return df_out


class ModelFactory:
    """Initializes and manages the specific algorithms."""

    @staticmethod
    def get_model(model_type: str, params: Dict, feature_cols: List[str] = None):
        model_type = model_type.lower()

        if "lightgbm" in model_type or "lgbm" in model_type:
            if lgb is None:
                raise ImportError("LightGBM not installed")
            return LightGBMWrapper(params)

        elif "xgboost" in model_type or "xgb" in model_type:
            if xgb is None:
                raise ImportError("XGBoost not installed")
            return XGBoostWrapper(params)

        elif "prophet" in model_type:
            if Prophet is None:
                raise ImportError("Prophet not installed")
            return ProphetWrapper(params)

        elif "svm" in model_type or "svr" in model_type:
            if SVR is None:
                raise ImportError("sklearn not installed")
            return SVMWrapper(params)

        else:
            raise ValueError(f"Unknown model type suggested: {model_type}")


class LightGBMWrapper:
    def __init__(self, params):
        self.params = params
        self.model = lgb.LGBMRegressor(**params)
        self.feature_names = []

    def fit(self, X, y):
        self.feature_names = list(X.columns)
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)


class XGBoostWrapper:
    def __init__(self, params):
        self.params = params
        self.model = xgb.XGBRegressor(**params)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)


class SVMWrapper:
    def __init__(self, params):
        # SVR needs scaling
        self.params = params
        self.pipe = Pipeline([("scaler", StandardScaler()), ("svr", SVR(**params))])

    def fit(self, X, y):
        self.pipe.fit(X, y)

    def predict(self, X):
        return self.pipe.predict(X)


class ProphetWrapper:
    def __init__(self, params):
        self.params = params
        self.model = Prophet(**params)
        self.regressors = []

    def fit(self, X, y):
        # Reconstruct DataFrame for Prophet
        df = X.copy()
        df["y"] = y
        # Identify ds column (usually parsed out in X, need to ensure it exists)
        # In this architecture, X might be all numeric features.
        # We need the date column back. This is a slight interface mismatch
        # that needs handling in the main Forecaster class.
        # For now, we assume 'ds' is passed in X or we error.
        if "ds" not in df.columns:
            raise ValueError("Prophet requires 'ds' column in features")

        for col in df.columns:
            if col not in ["ds", "y"]:
                self.model.add_regressor(col)
                self.regressors.append(col)

        self.model.fit(df)

    def predict(self, X):
        if "ds" not in X.columns:
            raise ValueError("Prophet requires 'ds' column")
        return self.model.predict(X)["yhat"].values


class NeuroForecaster:
    def __init__(self, llm_function: Callable[[str], str], verbose=True):
        """
        args:
            llm_function: A function that takes a string (prompt) and returns a string (response).
        """
        self.llm_func = llm_function
        self.model_wrapper = None
        self.feature_gen = None
        self.date_col = None
        self.y_col = None
        self.regressors = []
        self.verbose = verbose
        self.suggestion = None

    def _construct_prompt(self, profile: Dict) -> str:
        return f"""
You are an expert Time Series Forecasting AutoML system. 
I will provide statistics about a dataset. You must recommend the BEST algorithm 
and a specific set of hyperparameters.

**Dataset Statistics:**
- Length: {profile["n_samples"]} rows
- Range: {profile["start_date"]} to {profile["end_date"]}
- Trend: {profile["trend"]}
- Seasonality: {profile["seasonality"]}
- Volatility: {profile["volatility"]}
- Missing/Zero values: {profile["missing_values"]} missing, {profile["zero_values"]} zeros.
- Available Regressors: {profile["regressors_available"]}

**Candidate Algorithms:**
1. LightGBM (Good for large data, complex patterns, supports lags)
2. XGBoost (Robust, handles non-linearities well)
3. Prophet (Best for strong seasonality, changepoints, handles holidays)
4. SVM (SVR) (Good for smaller, noisy datasets with clear trends)

**Response Format:**
Return ONLY a valid JSON object with this structure:
{{
    "model": "one of [lightgbm, xgboost, prophet, svm]",
    "params": {{ key: value pairs for the chosen library }},
    "feature_strategy": {{
        "lags": boolean,
        "rolling": boolean,
        "date_parts": boolean
    }},
    "reasoning": "Short explanation why"
}}

Do not return code blocks or markdown. Just the raw JSON string.
        """

    def fit(self, df: pd.DataFrame, date_col: str, y_col: str, regressors: List[str] = None):
        self.date_col = date_col
        self.y_col = y_col
        self.regressors = regressors if regressors else []

        # 1. Profile Data
        if self.verbose:
            print("Analyzing dataset profile...")
        profile = DataProfiler.analyze(df, date_col, y_col, regressors)

        # 2. Ask LLM
        prompt = self._construct_prompt(profile)
        if self.verbose:
            print("Consulting LLM for model selection...")
        response_str = self.llm_func(prompt)

        try:
            # Clean markdown code blocks if present
            cleaned_resp = re.sub(r"```json|```", "", response_str).strip()
            resp_json = json.loads(cleaned_resp)

            self.suggestion = ModelSuggestion(
                model_type=resp_json["model"],
                params=resp_json.get("params", {}),
                reasoning=resp_json.get("reasoning", ""),
                feature_strategy=resp_json.get("feature_strategy", {}),
            )

            if self.verbose:
                print(f"LLM Selected: {self.suggestion.model_type}")
                print(f"Reasoning: {self.suggestion.reasoning}")
                print(f"Params: {self.suggestion.params}")

        except Exception as e:
            print(f"Error parsing LLM response: {e}. Falling back to Prophet default.")
            self.suggestion = ModelSuggestion(
                model_type="prophet",
                params={},
                reasoning="Fallback due to parse error",
                feature_strategy={"lags": False, "rolling": False, "date_parts": False},
            )

        # 3. Feature Engineering
        # If Prophet, we skip heavy manual features as it handles them,
        # unless explicitly requested, but typically Prophet expects raw 'ds'
        is_prophet = "prophet" in self.suggestion.model_type.lower()

        self.feature_gen = FeatureGenerator(date_col, y_col, self.suggestion.feature_strategy)

        X_train_full = self.feature_gen.transform(df)

        # Prepare X and y
        if is_prophet:
            # Prophet expects 'ds' and 'y'
            X_train = X_train_full.rename(columns={date_col: "ds"})
            y_train = df[
                y_col
            ].values  # Prophet ignores this y argument usually, but standardizing API
            # Keep regressors + generated features
            keep_cols = (
                ["ds"]
                + self.regressors
                + [c for c in X_train.columns if c not in [date_col, y_col, "ds"]]
            )
            X_train = X_train[keep_cols]
        else:
            # Tree/SVM models: Drop date column (unless converted to numeric), drop target
            drop_cols = [date_col, y_col]
            X_train = X_train_full.drop(
                columns=[c for c in drop_cols if c in X_train_full.columns]
            )
            y_train = X_train_full[y_col].values

        # 4. Train
        if self.verbose:
            print(f"Training {self.suggestion.model_type}...")
        self.model_wrapper = ModelFactory.get_model(
            self.suggestion.model_type, self.suggestion.params
        )
        self.model_wrapper.fit(X_train, y_train)

        if self.verbose:
            print("Training complete.")

    def predict(self, df_future: pd.DataFrame) -> np.array:
        """
        df_future must contain date_col and any regressors.
        If lags are used, df_future must be appended to history to calculate them,
        or provided with pre-calculated lags.

        *Simplification for this demo:* We assume df_future has necessary columns
        or we are doing a direct horizon forecast where we know the calendar features.
        For lags, in a real recursive loop, we'd update row by row.
        Here we apply feature gen assuming context exists.
        """
        is_prophet = "prophet" in self.suggestion.model_type.lower()

        X_test_full = self.feature_gen.transform(df_future, is_training=False)

        if is_prophet:
            X_test = X_test_full.rename(columns={self.date_col: "ds"})
            keep_cols = (
                ["ds"]
                + self.regressors
                + [c for c in X_test.columns if c not in [self.date_col, self.y_col, "ds"]]
            )
            # Ensure intersection with training columns
            valid_cols = [
                c for c in keep_cols if c in X_test.columns
            ]  # logic needs strict alignment in prod
            X_test = X_test[valid_cols]
        else:
            drop_cols = [self.date_col, self.y_col]
            # Ensure we only keep columns used in training
            try:
                model_feats = self.model_wrapper.feature_names
                X_test = X_test_full[model_feats]
            except AttributeError:
                # For SVM/XGB wrapper if feature names aren't stored strictly
                X_test = X_test_full.drop(
                    columns=[c for c in drop_cols if c in X_test_full.columns]
                )

        return self.model_wrapper.predict(X_test)


# ==========================================
# Example Usage
# ==========================================
if __name__ == "__main__":
    # 1. Create Dummy Data
    dates = pd.date_range(start="2022-01-01", periods=365, freq="D")
    # Linear trend + Weekly seasonality + Noise
    values = (
        np.linspace(10, 50, 365)
        + 10 * np.sin(np.arange(365) * (2 * np.pi / 7))
        + np.random.normal(0, 2, 365)
    )

    df = pd.DataFrame({"date": dates, "sales": values})

    # 2. Define a Mock LLM Function (Replace this with your OpenAI/Gemini API call)
    def mock_llm_brain(prompt_text):
        # In reality, you would do:
        # response = client.chat.completions.create(messages=[{"role": "user", "content": prompt_text}])
        # return response.choices[0].message.content

        print("\n--- PROMPT SENT TO LLM ---")
        print(prompt_text[:300] + "...")
        print("--------------------------\n")

        # Simulating a decision based on the prompt data
        return json.dumps(
            {
                "model": "lightgbm",
                "params": {"n_estimators": 100, "learning_rate": 0.05, "num_leaves": 31},
                "feature_strategy": {"lags": True, "rolling": True, "date_parts": True},
                "reasoning": "The data shows weekly seasonality and a stable trend. LightGBM handles the generated lag features efficiently for this sample size.",
            }
        )

    # 3. Initialize and Run
    nf = NeuroForecaster(llm_function=mock_llm_brain)

    # Split for demo
    train_df = df.iloc[:-30]
    test_df = df.iloc[-30:]

    nf.fit(train_df, date_col="date", y_col="sales")

    # Predict
    preds = nf.predict(test_df)

    print("\nForecast Sample:", preds[:5])
    print("Actual Sample:", test_df["sales"].values[:5])
