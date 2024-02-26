import pandas as pd
import numpy as np
import time

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler

from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

from logger import log


class StopScriptException(Exception):
    pass


class PreProcess:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def keep_cols(self, cols: list) -> pd.DataFrame:
        return self.df[cols]

    def drop_cols(self, cols: list) -> pd.DataFrame:
        return self.df.drop(columns=cols)

    def checking_data(self) -> pd.DataFrame:
        return pd.DataFrame({
            "Features": self.df.columns,
            "Count Rows": self.df.count(),
            "Data Types": self.df.dtypes,
            "Missing Values (%)": (self.df.isna().sum() * 100) / len(self.df),
            "Unique Values": self.df.nunique(),
            "Sample Values": self.df.apply(lambda col: col.unique()),
        })
    
    def fill_null(self, method: str = "median") -> pd.DataFrame:
        num_cols = self.df.select_dtypes(include=["number"]).columns.tolist()
        log(f"Filling Null values for numerical columns ({', '.join(num_cols)}).")

        if method == "median":
            for col in num_cols:
                self.df.loc[:, col] = self.df[col].fillna(self.df[col].median())
            log(f"Finished filling Null values with the '{method.title()}'.")
            return self.df
        elif method == "mean":
            for col in num_cols:
                self.df.loc[:, col] = self.df[col].fillna(self.df[col].mean())
            log(f"Finished filling Null values with the '{method.title()}'.")
            return self.df
        else:
            raise StopScriptException("Invalid method! Please choose either 'median' or 'mean'.")
        
    def find_missing_intervals_message(self, dt_col: str="dt_iso") -> str:
        self.df[dt_col] = pd.to_datetime(self.df[dt_col], errors='coerce')
        unique_datetimes = self.df.sort_values(dt_col)[dt_col].unique()
        missing_intervals = []

        # Loop through the datetime array to check for missing hour intervals
        for i in range(1, len(unique_datetimes)):
            time_diff = (unique_datetimes[i] - unique_datetimes[i - 1]) / pd.Timedelta(hours=1)

            if time_diff > 1:
                missing_intervals.append((unique_datetimes[i - 1], unique_datetimes[i]))

        if not missing_intervals:
            log("No missing hour intervals.")
        else:
            raise StopScriptException("Missing hour intervals:\n" + "\n".join([f"From {interval[0]} to {interval[1]}" for interval in missing_intervals]))
        

class PreForecast:
    def __init__(self, df: pd.DataFrame, target_cols: list, models: dict, hours_to_forecast: int):
        self.df = df
        self.target_cols = target_cols
        self.models = models
        self.hours_to_forecast = hours_to_forecast

    def feature_engineering(self, cat_cols: list, dt_col: str="dt_iso") -> pd.DataFrame:
        log(f"Forecast: Extracting the necessary information from the {dt_col} column.")

        self.df["month"] = self.df[dt_col].dt.month
        self.df["week"] = self.df[dt_col].dt.isocalendar().week
        self.df["day"] = self.df[dt_col].dt.day
        self.df["hour"] = self.df[dt_col].dt.hour

        encoder = OneHotEncoder()
        encoded_features = encoder.fit_transform(self.df[cat_cols])

        encoded_df = pd.DataFrame(encoded_features.toarray(), columns=encoder.get_feature_names_out(cat_cols))
        self.df = pd.concat([self.df, encoded_df], axis=1)
        self.df = self.df.drop(columns=cat_cols + [dt_col])
        return self.df
    
    def train_forecast_data(self, dt_col: str="dt", forecast: bool=False) -> pd.DataFrame:
        lookback = self.hours_to_forecast * 2
        self.df = self.df.sort_values(dt_col, ascending=False)

        mssg = (
            "Forecast: Preparing the data to obtain "
            "the best model."
        )
        if forecast:
            empty_df = pd.DataFrame(
                index=range(self.hours_to_forecast),
                columns=self.df.columns,
            )
            self.df = pd.concat([empty_df, self.df], ignore_index=True)
            mssg = (
                f"Forecast: Preparing the data "
                f"for forecasting {self.hours_to_forecast} hours ahead."
            )

        log(mssg)
        for col in self.df.columns:
            if col != dt_col:
                new_cols = [
                    self.df[col].shift(periods=-lags).rename(f"D-{lags}_{col}")
                    for lags in range(
                        self.hours_to_forecast, (lookback + self.hours_to_forecast)
                    )
                ]
                self.df = pd.concat([self.df] + new_cols, axis=1)

        col_features = [col for col in self.df.columns if col.startswith("D-")]
        if forecast:
            self.df = self.df[col_features]
            self.df = self.df.iloc[:self.hours_to_forecast]
            self.df = self.df.sort_index(ascending=False)
        else:
            self.df = self.df[[dt_col] + col_features]
            self.df = self.df.sort_values(dt_col)

        return self.df
    
    def best_models(self, df_feature: pd.DataFrame, dt_col: str="dt") -> pd.DataFrame:
        log(f"Forecast: Starting Process to Get Best Models for Each Target Columns ({', '.join(self.target_cols)}).")

        def evaluate_model_forecast(y_test, pred, runtime):
            def format_time(seconds) -> str:
                hours, remainder = divmod(seconds, 3600)
                minutes, seconds = divmod(remainder, 60)
                return "{:02}:{:02}:{:02}".format(int(hours), int(minutes), int(seconds))
            
            mae = round(mean_absolute_error(y_test, pred), 2)
            ae = abs((y_test - pred) / y_test)
            mape = round((np.sum(ae) / len(ae)) * 100, 2)
            wmape = round(sum(abs(y_test - pred)) / sum(y_test) * 100, 2)
            formatted_runtime = format_time(runtime)
            return mae, mape, wmape, formatted_runtime

        best_models = pd.DataFrame()

        for col in self.target_cols:
            log(f"  > Column '{col}'")
            pre_train = self.df.copy()

            pre_train.insert(1, col, df_feature[col])
            pre_train = pre_train.dropna()

            train = pre_train.iloc[:-self.hours_to_forecast]
            test = pre_train.iloc[-self.hours_to_forecast:]

            X_train, y_train = train.drop(columns=[dt_col, col]), train[col]
            X_test, y_test = test.drop(columns=[dt_col, col]), test[col]

            scaler = MinMaxScaler()
            X_train_scale = scaler.fit_transform(X_train)
            X_test_scale = scaler.transform(X_test)

            for model_name, model in self.models.items():
                log(f"       with Model '{model_name}'")
                start_time = time.time()

                m = model
                m.fit(X_train_scale, y_train)
                pred = m.predict(X_test_scale)

                runtime = time.time() - start_time
                scores = evaluate_model_forecast(y_test, pred, runtime)
                row = pd.DataFrame(
                    {
                        "Target": col,
                        "Model": model_name,
                        "MAE": scores[0],
                        "MAPE": scores[1],
                        "WMAPE": scores[2],
                        "Runtime": scores[3],
                    },
                    index=[0],
                )
                best_models = pd.concat([best_models, row], ignore_index=True)

        best_models["Runtime"] = pd.to_timedelta(best_models["Runtime"])
        best_models = best_models.sort_values(by=["Target", "WMAPE", "Runtime"])

        return best_models
    

class Forecast:
    def __init__(
            self,
            best_models: pd.DataFrame,
            df_train: pd.DataFrame,
            df_feature: pd.DataFrame,
            df_forecast: pd.DataFrame,
            target_cols: list,
            models: dict,
            hours_to_forecast: int,
            last_dt: int,
            timezone: int,
            interval: int=3600,
            dt_cols: list=["dt", "dt_iso"]
    ):
        self.best_models = best_models
        self.df_train = df_train
        self.df_feature = df_feature
        self.df_forecast = df_forecast
        self.target_cols = target_cols
        self.models = models
        self.hours_to_forecast = hours_to_forecast
        self.last_dt = last_dt
        self.timezone = timezone
        self.interval = interval
        self.dt_cols = dt_cols

    def forecasting(self) -> pd.DataFrame:
        log(f"Forecast: Starting Forecast with Best Model for Each Target Columns ({', '.join(self.target_cols)}).")
        best_models = self.best_models.groupby("Target").first().reset_index()
        models = dict(zip(best_models["Target"], best_models["Model"]))

        forecast = pd.DataFrame(columns=self.dt_cols)
        for col in self.target_cols:
            log(f"  > Column '{col}' -> '{models.get(col)}'")
            train = self.df_train.copy()
            forecast_ = self.df_forecast.copy()

            train.insert(1, col, self.df_feature[col])
            train = train.dropna()

            X_train = train.drop(columns=[self.dt_cols[0], col], axis=1)
            y_train = train[col]

            scaler = MinMaxScaler()
            X_train_scale = scaler.fit_transform(X_train)
            forecast_scale = scaler.transform(forecast_)

            m = self.models.get(models.get(col))
            m.fit(X_train_scale, y_train)
            forecast[col] = m.predict(forecast_scale)
            forecast[col] = forecast[col].astype(float).round(
                2 if col in self.target_cols[:-2] else 0
            )
            
        forecast[self.dt_cols[0]] = [self.last_dt + (self.interval * n) for n in range(1, self.hours_to_forecast + 1)]
        forecast[self.dt_cols[1]] = pd.to_datetime(forecast[self.dt_cols[0]] + self.timezone, unit="s")

        return forecast
    

class PreClassify:
    def __init__(self, df: pd.DataFrame, feature_cols: list, target_col: str):
        self.df = df
        self.feature_cols = feature_cols
        self.target_col = target_col

    def feature_engineering(self, dt_col: str="dt_iso") -> pd.DataFrame:
        log(f"Classification: Extracting the necessary information from the {dt_col} column.")
        self.df["month"] = self.df[dt_col].dt.month
        self.df["week"] = self.df[dt_col].dt.isocalendar().week
        self.df["day"] = self.df[dt_col].dt.day
        self.df["hour"] = self.df[dt_col].dt.hour

        cols = self.feature_cols + [self.target_col]
        self.df = self.df[cols]
        return self.df
    
    def handle_outliers(self) -> pd.DataFrame:
        log(f"Classification: Staring to Handle Outliers From Freature Columns {', '.join(self.feature_cols)}")
        before = self.df.shape[0]
        for col in self.feature_cols:
            q1 = self.df[col].quantile(.25)
            q3 = self.df[col].quantile(.75)
            iqr = q3 - q1

            low_limit = q1 - (1.5 * iqr)
            high_limit = q3 + (1.5 * iqr)

            filtered = ((self.df[col] >= low_limit) & (self.df[col] <= high_limit))
            self.df = self.df[filtered]
        after = self.df.shape[0]

        log(f"Classification: Reduced by {round((100 - ((after / before) * 100)), 2)}% from previously {before} to {after} data.")
        return self.df
    
    def handle_imbalance(self, threshold_percent: float=10, dt_col: str="dt") -> pd.DataFrame:
        # Determine the minority class
        class_counts = self.df[self.target_col].value_counts()
        minority_class = class_counts.idxmin()
        minority_count = class_counts.min()


        # Calculate the threshold in terms of the number of samples
        total_samples = len(self.df)
        threshold_samples = int(threshold_percent * total_samples / 100)

        # Check if the number of samples of the minority class is less than the threshold
        if minority_count < threshold_samples:
            log(f"Classification: Handling imbalance for class '{minority_class}' with count {minority_count}.")

            # Separate features and labels
            X = self.df.drop(columns=[self.target_col])
            y = self.df[self.target_col]

            # Apply RandomOverSampler for oversampling
            ros = RandomOverSampler(random_state=42)
            X_resampled, y_resampled = ros.fit_resample(X, y)

            # Create a new DataFrame from the oversampling results
            df_resampled = pd.concat([X_resampled, y_resampled], axis=1)
            df_resampled = df_resampled.drop(columns=dt_col)
            return df_resampled
        else:
            log("Classification: Skipping imbalance handling as minority class count exceeds threshold.")
            return self.df
        
    def best_models(self, models: dict) -> pd.DataFrame:
        log(f"Classification: Starting Process to Get Best Model.")

        def evaluate_model_classification(y_test, pred, runtime):
            def format_time(seconds) -> str:
                hours, remainder = divmod(seconds, 3600)
                minutes, seconds = divmod(remainder, 60)
                return "{:02}:{:02}:{:02}".format(int(hours), int(minutes), int(seconds))
            
            accuracy = accuracy_score(y_test, pred)

            precision = precision_score(y_test, pred, average='weighted')
            recall = recall_score(y_test, pred, average='weighted')
            f1 = f1_score(y_test, pred, average='weighted')
            formatted_runtime = format_time(runtime)
            return accuracy, precision, recall, f1, formatted_runtime

        best_models = pd.DataFrame()
        pre_train = self.df.copy()

        # Initialize LabelEncoder
        label_encoder = LabelEncoder()
        pre_train[self.target_col] = label_encoder.fit_transform(pre_train[self.target_col])

        X = pre_train.drop(self.target_col, axis=1)
        y = pre_train[self.target_col]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler = MinMaxScaler()
        X_train_scale = scaler.fit_transform(X_train)
        X_test_scale = scaler.transform(X_test)

        for model_name, model in models.items():
            log(f"       with Model '{model_name}'")
            start_time = time.time()

            m = model
            m.fit(X_train_scale, y_train)
            pred = m.predict(X_test_scale)

            runtime = time.time() - start_time
            scores = evaluate_model_classification(y_test, pred, runtime)
            row = pd.DataFrame(
                {
                    "Model": model_name,
                    "Accuracy": scores[0],
                    "Precision": scores[1],
                    "Recall": scores[2],
                    "F1-score": scores[3],
                    "Runtime": scores[4],
                },
                index=[0],
            )
            best_models = pd.concat([best_models, row], ignore_index=True)

        best_models["Runtime"] = pd.to_timedelta(best_models["Runtime"])
        best_models = best_models.sort_values(
            by=["F1-score", "Runtime"], ascending=[False, True]
        )
        
        log(f"Classification: Finished Process to Get Best Model.")
        return best_models
    
class Classify:
    def __init__(self, df: pd.DataFrame, df_forecast: pd.DataFrame, feature_cols: list, target_col: str, models: dict, dt_cols: list=["dt", "dt_iso"]) -> pd.DataFrame:
        self.df = df
        self.df_forecast = df_forecast
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.models = models
        self.dt_cols = dt_cols

    def classification(self, best_models: pd.DataFrame, probability_class: str="Rain") -> pd.DataFrame:
        log(f"Classification: Start Process Classify with Best Model.")
        result = self.df_forecast.copy()
        models = best_models.iloc[[0]]

        result = result.drop(self.dt_cols[0], axis=1)
        result["month"] = result[self.dt_cols[1]].dt.month
        result["week"] = result[self.dt_cols[1]].dt.isocalendar().week
        result["day"] = result[self.dt_cols[1]].dt.day
        result["hour"] = result[self.dt_cols[1]].dt.hour
        result = result.drop(self.dt_cols[1], axis=1)

        # Initialize LabelEncoder
        label_encoder = LabelEncoder()
        self.df [self.target_col] = label_encoder.fit_transform(self.df [self.target_col])
        label_mapping  = dict(zip(
            label_encoder.classes_, label_encoder.transform(label_encoder.classes_)
        ))
        label = label_mapping[probability_class]

        X = self.df.drop(self.target_col, axis=1)
        y = self.df[self.target_col]

        scaler = MinMaxScaler()
        X_train_scale = scaler.fit_transform(X)
        classification = scaler.transform(result)

        log(f"       with Model '{models.iloc[0, 0]}'")
        m = self.models.get(models.iloc[0, 0])
        m.fit(X_train_scale, y)
        pred = m.predict(classification)
        pred_prob = m.predict_proba(classification)

        self.df_forecast["Weather"] = label_encoder.inverse_transform(pred)
        self.df_forecast[f"Prob. Of {probability_class}"] = (
            pd.Series(pred_prob[:, int(label)] * 100)
            .astype(float)
            .round(2)
            .map(lambda x: "{:.2f}%".format(x))
        )
        log(f"Classification: Finished Process Classify with Best Model.")
        return self.df_forecast
