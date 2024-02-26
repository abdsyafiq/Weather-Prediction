from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
import lightgbm
import xgboost

from main import WeatherPredictor

# MariaDB Configurations and Parameters
maria_conf = {
    "username": "username",
    "password": "password",
    "host": "host",
    "database": "database",
    "table": "table",
    "location": "location",
}

# Columns to Keep
keep_cols = [
    "dt", "dt_iso", "temp", "dew_point", "feels_like",
    "temp_min", "temp_max", "pressure", "humidity",
    "wind_speed", "wind_deg", "clouds_all", "weather_main",
]

# Target Columns for Forecasting
target_cols_forecast = [
    "temp", "dew_point", "feels_like", "temp_min",
    "temp_max", "pressure", "humidity",
]

# Models for Forecasting
models_forecast = {
    "XGBoost": xgboost.XGBRegressor(),
    "K-NearestNeighbor": KNeighborsRegressor(),
    "Linear": LinearRegression(),
    "Ridge": Ridge(),
    "Lasso": Lasso(),
    # "ElasticNet": ElasticNet(),
    "DecisionTree": DecisionTreeRegressor(),
    "AdaBoost": AdaBoostRegressor(),
}

# Feature Columns for Classification
feature_cols_classify = [
    "dt", "temp", "dew_point", "feels_like",
    "temp_min", "temp_max", "pressure", "humidity",
    "month", "week", "day", "hour"
]

# Target Column for Classification
target_col_classify = "weather_main"

# Models for Classification
models_classify = {
    "DecisionTree": DecisionTreeClassifier(),
    # "RandomForest": RandomForestClassifier(),
    "K-NearestNeighbors": KNeighborsClassifier(),
    "AdaBoost": AdaBoostClassifier(),
    "GradientBoost": GradientBoostingClassifier(),
    "XGBoost": xgboost.XGBClassifier(),
    "LightGBM": lightgbm.LGBMClassifier(),
    "NaiveBayes": GaussianNB(),
}

# Hours to Forecast
hours_to_forecast = 48

# Instantiate WeatherPredictor
weather_predictor = WeatherPredictor(
    maria_conf,
    keep_cols,
    target_cols_forecast,
    models_forecast,
    feature_cols_classify,
    target_col_classify,
    models_classify,
)

# Connect to Database
raw_data = weather_predictor.connect_to_database()

# Process Data
processed_data = weather_predictor.process_data(raw_data)

# Forecast
forecast_result = weather_predictor.forecast(processed_data, raw_data, hours_to_forecast)
forecast_result.to_csv("./result/weather_result_forecast.csv", index=False)

# Classification
classification_result = weather_predictor.classifation(processed_data, forecast_result)
classification_result.to_csv("./result/weather_result_classification.csv", index=False)
