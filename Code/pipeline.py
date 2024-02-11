from typing import Literal
from dataset import Dataset
import sklearn.model_selection
import sklearn.preprocessing
import sklearn.pipeline


class Pipeline:
    def __init__(self, crypto_tick: Literal["btc", "ltc", "eth"]) -> None:
        self.tick = crypto_tick
        match self.tick:
            case "btc":
                self.data = Dataset.get_btc_data()
            case "ltc":
                self.data = Dataset.get_ltc_data()
            case "eth":
                self.data = Dataset.get_eth_data()
        self.data_1d_shift = None
        self.data_5d_shift = None
        self.data_10d_shift = None

    def set_beginning(self, start_date: str):
        self.data = self.data.loc[start_date:]
        return self

    def preprocess_dataset(self):
        # Filling crypto wiki searches with zeros as the data started to be collected later
        self.data['Wiki_crypto_search'] = self.data['Wiki_crypto_search'].fillna(
            0)
        # Forward filling all the indexes into weekends and other day when the maret was closed
        self.data[['Close_^DJI', 'Close_^GSPC', 'Close_GC=F', 'Close_^VIX', 'Close_^IXIC',
                   'Close_SMH', 'Close_VGT', 'Close_XSD', 'Close_IYW', 'Close_FTEC', 'Close_IGV',
                   'Close_QQQ', 'USD_EUR_rate']] = self.data[['Close_^DJI', 'Close_^GSPC', 'Close_GC=F', 'Close_^VIX', 'Close_^IXIC',
                                                              'Close_SMH', 'Close_VGT', 'Close_XSD', 'Close_IYW', 'Close_FTEC', 'Close_IGV',
                                                              'Close_QQQ', 'USD_EUR_rate']].ffill()
        match self.tick:
            case "btc":
                self.data['Wiki_btc_search'] = self.data['Wiki_btc_search'].fillna(
                    0)
                # also filling with zeros as the metric was not present at that time
                self.data["BTC / Capitalization, market, estimated supply, USD"] = self.data[
                    "BTC / Capitalization, market, estimated supply, USD"].fillna(0)
            case "ltc":
                self.data['Wiki_ltc_search'] = self.data['Wiki_ltc_search'].fillna(
                    0)
                self.data["LTC / Capitalization, market, estimated supply, USD"] = self.data[
                    "LTC / Capitalization, market, estimated supply, USD"].fillna(0)
            case "eth":
                self.data['Wiki_eth_search'] = self.data['Wiki_eth_search'].fillna(
                    0)
                self.data["ETH / Capitalization, market, estimated supply, USD"] = self.data[
                    "ETH / Capitalization, market, estimated supply, USD"].fillna(0)
                # cutting the end where the PoS consensus mechanism starts
                self.data = self.data.loc[:"2022-09-15"]
                # dropping the beginning where the moving averages are not present
                self.data = self.data.dropna()
        return self

    def shift_target(self):
        # 1 day forecasting
        self.data_1d_shift = self.data.copy()
        self.data_1d_shift.iloc[:, -
                                1] = self.data_1d_shift.iloc[:, -1].shift(-1)
        self.data_1d_shift = self.data_1d_shift.dropna()
        # 5 day forecasting
        self.data_5d_shift = self.data.copy()
        self.data_5d_shift.iloc[:, -
                                1] = self.data_5d_shift.iloc[:, -1].shift(-5)
        self.data_5d_shift = self.data_5d_shift.dropna()
        # 10 day forecasting
        self.data_10d_shift = self.data.copy()
        self.data_10d_shift.iloc[:, -
                                 1] = self.data_10d_shift.iloc[:, -1].shift(-10)
        self.data_10d_shift = self.data_10d_shift.dropna()
        return self

    @staticmethod
    def split_train_test(data):
        train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
            data.iloc[:, :-1], data.iloc[:, -1], test_size=0.1, random_state=42, shuffle=False)
        return train_data, test_data, train_target, test_target

    @staticmethod
    def assembly_pipeline(estimator, dim_reducer=None):
        scaler = sklearn.preprocessing.RobustScaler()
        pipeline = sklearn.pipeline.Pipeline([("scaler", scaler),
                                              ("dim_reducer", dim_reducer), ("estimator", estimator)])
        return pipeline

    @staticmethod
    def fit_grid_search(train_data, train_target, pipeline, parameter_grid):
        scoring = {"RMSE" : "neg_root_mean_squared_error",
                   "MAE" : "neg_mean_absolute_error",
                   "MAPE" : "neg_mean_absolute_percentage_error"}
        ts_split = sklearn.model_selection.TimeSeriesSplit(n_splits=3)
        model = sklearn.model_selection.GridSearchCV(
            pipeline, param_grid = parameter_grid,
            cv=ts_split, scoring=scoring, refit = "RMSE",
            verbose=4, n_jobs=-1).fit(train_data, train_target)
        return model
