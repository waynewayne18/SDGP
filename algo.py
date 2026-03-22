import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor


class Algo:
    def __init__(self,forecasting_for, training_weeks=6):
        # training_weeks controls both the test set size and the forecast length
        self.training_weeks = training_weeks
        self.forecasting_for = forecasting_for
        # ── Load raw CSV files ────────────────────────────────────────────────
        # Coffee file has a header row that needs to be skipped
        self.croissant = pd.read_csv("Croissant_Sales.csv", header=0)
        self.coffee    = pd.read_csv("Coffee_Sales.csv", skiprows=1, header=0)

        self.croissant.columns = ["Date", "Croissant"]
        self.coffee.columns    = ["Date", "Cappuccino", "Americano"]

        croi_dates = self.croissant["Date"]
        coff_dates = self.coffee["Date"]

        # ── Build a unified dataframe with one row per product per day ────────
        # Each product gets a "product" label so they can be grouped later
        self.croi = pd.DataFrame({
            "date":    pd.to_datetime(croi_dates, dayfirst=True),
            "value":   self.croissant["Croissant"],
            "product": "croi"
        })
        self.cappy = pd.DataFrame({
            "date":    pd.to_datetime(coff_dates, dayfirst=True),
            "value":   self.coffee["Cappuccino"],
            "product": "cappy"
        })
        self.ameri = pd.DataFrame({
            "date":    pd.to_datetime(coff_dates, dayfirst=True),
            "value":   self.coffee["Americano"],
            "product": "ameri"
        })

        # Stack all three products into one dataframe
        self.all_products = pd.concat(
            [self.croi, self.cappy, self.ameri], ignore_index=True
        )

        # Encode product names as integers — required by XGBoost
        self.all_products["product_cat"] = (
            self.all_products["product"].astype("category").cat.codes
        )

        # Feature list used for both training and prediction — must stay consistent
        self.features = [
            "day_of_week", "day_of_month", "month", "quarter",
            "is_weekend", "lag_1", "lag_7", "lag_14", "lag_30",
            "rolling_7", "rolling_14", "rolling_30"
        ]

    def featureCreation(self, df):
        df = df.copy()

        # Calendar features — help the model learn day-of-week and seasonal patterns
        df["day_of_week"]  = df["date"].dt.dayofweek
        df["day_of_month"] = df["date"].dt.day
        df["month"]        = df["date"].dt.month
        df["quarter"]      = df["date"].dt.quarter
        df["is_weekend"]   = df["day_of_week"].isin([5, 6]).astype(int)

        # Lag features — past sales values used as predictors for future sales
        # Grouped by product so lags do not bleed across different products
        df = df.sort_values(["product", "date"])
        df["lag_1"]  = df.groupby("product")["value"].shift(1)   # yesterday
        df["lag_7"]  = df.groupby("product")["value"].shift(7)   # same day last week
        df["lag_14"] = df.groupby("product")["value"].shift(14)  # two weeks ago
        df["lag_30"] = df.groupby("product")["value"].shift(30)  # one month ago

        # Rolling averages — smooth out daily noise and capture recent trends
        df["rolling_7"]  = df.groupby("product")["value"].transform(
            lambda x: x.rolling(7).mean()
        )
        df["rolling_14"] = df.groupby("product")["value"].transform(
            lambda x: x.rolling(14).mean()
        )
        df["rolling_30"] = df.groupby("product")["value"].transform(
            lambda x: x.rolling(30).mean()
        )

        # Drop rows where lag/rolling values could not be computed (start of the series)
        return df.dropna()

    def forecast(self, forecasting_for, days=None):
        # Default forecast length is based on the training window
        if days is None:
            days = self.training_weeks * 7
        forecast_days = forecasting_for * 7
        print(forecasting_for, "--------------------------------------------------")

        history = self.all_products.copy()
        all_forecasts = []

        for product in ["croi", "cappy", "ameri"]:
            product_history = (
                history[history["product"] == product]
                .copy()
                .sort_values("date")
                .reset_index(drop=True)
            )
            product_cat = product_history["product_cat"].iloc[0]
            future_predictions = []
            print (forecast_days)

            for day in range(1, forecast_days + 1):
                # Always predict one day beyond the last known date
                future_date   = product_history["date"].max() + pd.Timedelta(days=1)
                recent_values = product_history["value"].values

                # Build the feature row for this future date manually
                # Lag and rolling values are taken from the growing product_history
                row = {
                    "day_of_week":  future_date.dayofweek,
                    "day_of_month": future_date.day,
                    "month":        future_date.month,
                    "quarter":      future_date.quarter,
                    "is_weekend":   int(future_date.dayofweek in [5, 6]),
                    "lag_1":        recent_values[-1],
                    "lag_7":        recent_values[-7],
                    "lag_14":       recent_values[-14],
                    "lag_30":       recent_values[-30],
                    "rolling_7":    np.mean(recent_values[-7:]),
                    "rolling_14":   np.mean(recent_values[-14:]),
                    "rolling_30":   np.mean(recent_values[-30:]),
                }

                future = pd.DataFrame([row])[self.features]
                pred   = self.models[product].predict(future)[0]

                translator = {
                    "croi": "Croissant",
                    "cappy": "Cappuccino",
                    "ameri": "Americano"
                }
                future_predictions.append({
                    "date":     future_date,
                    "product":  translator[product],
                    "forecast": round(pred, 0)
                })

                # Append the prediction back into history so the next iteration
                # can use it as a lag value — this is called recursive forecasting
                new_row = pd.DataFrame([{
                    "date":        future_date,
                    "value":       pred,
                    "product":     product,
                    "product_cat": product_cat
                }])
                product_history = pd.concat(
                    [product_history, new_row], ignore_index=True
                )

            all_forecasts.extend(future_predictions)

        return pd.DataFrame(all_forecasts)

    def Predictor(self):
        maes = []
        self.all_products = self.featureCreation(self.all_products)
        self.models = {}

        # Test set = last training_weeks * 7 days; everything before is training data
        training_days = self.training_weeks * 7

        for product in ["croi", "cappy", "ameri"]:
            product_data = (
                self.all_products[self.all_products["product"] == product]
                .sort_values("date")
                .reset_index(drop=True)
            )

            # Calculate the split point — test set is the most recent N days
            split = len(product_data) - training_days
            if split <= 30:
                # Fall back to 80/20 split if there is not enough training data
                split = int(len(product_data) * 0.8)

            train_X = product_data[self.features][:split]
            test_X  = product_data[self.features][split:]
            train_y = product_data["value"][:split]
            test_y  = product_data["value"][split:]

            # ── XGBoost model configuration ───────────────────────────────────
            # early_stopping_rounds halts training if test error stops improving,
            # which prevents the model from overfitting to the training data
            model = XGBRegressor(
                n_estimators=1000,       # maximum number of trees
                learning_rate=0.01,      # small step size for more stable learning
                max_depth=5,             # limits tree complexity
                subsample=0.8,           # uses 80% of rows per tree — adds randomness
                colsample_bytree=0.8,    # uses 80% of features per tree — adds randomness
                early_stopping_rounds=50
            )
            model.fit(
                train_X, train_y,
                eval_set=[(test_X, test_y)],
                verbose=False
            )

            # Evaluate on the test set using MAE (Mean Absolute Error)
            # MAE = average number of units the model is off per day
            predictions = model.predict(test_X)
            mae = mean_absolute_error(test_y, predictions)

            translator = {
                "croi": "Croissant",
                "cappy": "Cappuccino",
                "ameri": "Americano"
            }
            print(f"{translator[product]} MAE: {mae:.2f} units")
            maes.append(mae)

            # Store the trained model so forecast() can use it later
            self.models[product] = model

        return maes


if __name__ == "__main__":
    algo = Algo(training_weeks=6)
    algo.Predictor()
    forecast_df = algo.forecast()
    print(forecast_df)