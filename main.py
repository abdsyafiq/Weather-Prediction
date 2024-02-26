import pandas as pd

from databases.db import MariaDBConnector
from model.process import PreProcess
from model.process import PreForecast
from model.process import Forecast
from model.process import PreClassify
from model.process import Classify

import warnings
warnings.filterwarnings("ignore")


class WeatherPredictor:
    def __init__(
            self,
            maria_conf,
            keep_cols,
            target_cols_forecast,
            models_forecast,
            feature_cols_classify,
            target_col_classify,
            models_classify,
    ):
        self.maria_conf = maria_conf
        self.keep_cols = keep_cols
        self.target_cols_forecast = target_cols_forecast
        self.models_forecast = models_forecast
        self.feature_cols_classify = feature_cols_classify
        self.target_col_classify = target_col_classify
        self.models_classify = models_classify

    def connect_to_database(self):
        conn = MariaDBConnector(**self.maria_conf)
        conn.connect()
        data = conn.fetch_data()
        raw_df = pd.DataFrame(data[1], columns=data[0])
        conn.close()
        return raw_df
    
    def process_data(self, raw_df):
        PreProcess(raw_df).find_missing_intervals_message()
        df = PreProcess(raw_df).keep_cols(self.keep_cols)
        df = PreProcess(df).fill_null("median")
        return df
    
    def forecast(self, df, raw_df, hours_to_forecast):
        # Prepare Data for Forecasting
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        df_feature = PreForecast(df, self.target_cols_forecast, self.models_forecast, hours_to_forecast).feature_engineering(cat_cols)
        df_train = PreForecast(df_feature, self.target_cols_forecast, self.models_forecast, hours_to_forecast).train_forecast_data()
        df_forecast = PreForecast(df_feature, self.target_cols_forecast, self.models_forecast, hours_to_forecast).train_forecast_data(forecast=True)

        # Get Best Models for Each Target Columns
        best_models = PreForecast(df_train, self.target_cols_forecast, self.models_forecast, hours_to_forecast).best_models(df_feature)
        best_models.to_csv("./result/best_models_forecast.csv", index=False)

        # Forecasting for Each Target Columns
        last_dt = raw_df.iloc[-1, 1]
        timezone = raw_df.iloc[-1, 3]
        forecast = Forecast(best_models, df_train, df_feature, df_forecast, self.target_cols_forecast, self.models_forecast, hours_to_forecast, last_dt, timezone).forecasting()
        return forecast
    
    def classifation(self, df, forecast):
        # Prepare Data for Classification
        df_feature = PreClassify(df, self.feature_cols_classify, self.target_col_classify).feature_engineering()
        df_cleaned = PreClassify(df_feature, self.feature_cols_classify, self.target_col_classify).handle_outliers()
        df_resampled = PreClassify(df_cleaned, self.feature_cols_classify, self.target_col_classify).handle_imbalance()

        # Get Best Models
        best_models = PreClassify(df_resampled, self.feature_cols_classify, self.target_col_classify).best_models(self.models_classify)
        best_models.to_csv("./result/best_models_classification.csv", index=False)

        # Classification
        result = Classify(df_resampled, forecast, self.feature_cols_classify, self.target_col_classify, self.models_classify).classification(best_models)
        return result
