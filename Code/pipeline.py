from typing import Literal
from dataset import Dataset
import sklearn.model_selection
import sklearn.preprocessing
import sklearn.pipeline
import tensorflow as tf
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


class PCATransformer(BaseEstimator, TransformerMixin):
    def __init__(self, pca):
        self.pca = pca

    def fit(self, X, y=None):
        self.pca.fit(X)
        return self

    def transform(self, X):
        # Downsample to n_components
        X_pca = self.pca.transform(X)
        # Upsample back to original dimensions
        X_restored = self.pca.inverse_transform(X_pca)
        return X_restored


class ReshapeTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, new_shape):
        self.new_shape = new_shape

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.reshape(self.new_shape)


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
    def split_train_test(data, pandas=True):
        if pandas:
            train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
                data.iloc[:, :-1], data.iloc[:, -1], test_size=0.1, random_state=42, shuffle=False)
        else:
            train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
                data[0], data[1], test_size=0.1, random_state=42, shuffle=False)
        return train_data, test_data, train_target, test_target

    @staticmethod
    def assembly_pipeline(estimator, dim_reducer, shape_change=False):
        scaler = sklearn.preprocessing.RobustScaler(unit_variance=True)
        if dim_reducer is not None:
            denoiser = PCATransformer(dim_reducer)
            scaler_2 = sklearn.preprocessing.StandardScaler()
        else:
            denoiser = dim_reducer
            scaler_2 = dim_reducer
        unpacker = None
        packer = None
        if shape_change is not False:
            unpacker = ReshapeTransformer(shape_change[0])
            packer = ReshapeTransformer(shape_change[1])
        pipeline = sklearn.pipeline.Pipeline([("pack_down", unpacker),
                                              ("scaler", scaler),
                                              ("denoiser", denoiser),
                                              ("scaler 2", scaler_2),
                                              ("pack_up,", packer),
                                              ("estimator", estimator)])

        return pipeline

    @staticmethod
    def fit_grid_search(train_data, train_target, pipeline, parameter_grid, n_jobs=-1):
        scoring = {"RMSE": "neg_root_mean_squared_error",
                   "MAE": "neg_mean_absolute_error",
                   "MAPE": "neg_mean_absolute_percentage_error"}
        ts_split = sklearn.model_selection.TimeSeriesSplit(n_splits=2)
        model = sklearn.model_selection.GridSearchCV(
            pipeline, param_grid=parameter_grid,
            cv=ts_split, scoring=scoring, refit="RMSE",
            verbose=0, n_jobs=n_jobs, error_score='raise').fit(train_data, train_target)
        return model

    @staticmethod
    def assembly_lstm(input_shape, units):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.LSTM(units, activation=None, input_shape=input_shape,
                                       return_sequences=True))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dropout(0.5))
        model.add(tf.keras.layers.ReLU())
        model.add(tf.keras.layers.LSTM(units, activation=None))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dropout(0.5))
        model.add(tf.keras.layers.ReLU())
        model.add(tf.keras.layers.Dense(1))
        inital_lr = 0.01
        lr_schedule = tf.keras.optimizers.schedules.CosineDecay(inital_lr, 2000)
        model.compile(optimizer=tf.keras.optimizers.Adam(
            learning_rate = lr_schedule, clipnorm = 1, clipvalue = 1), loss="mae", metrics=[Pipeline.mean_squared_percentage_error])
        return model

    @staticmethod
    def mean_squared_percentage_error(y_true, y_pred):
        return tf.reduce_mean(tf.square((y_true - y_pred) / tf.maximum(tf.abs(y_true), 1)))

    @staticmethod
    def create_lstm_input(data, target, lag_order, forecast_time = 1):
        X, Y = [], []
        data["BTC-USD"] = data["BTC-USD"].shift(forecast_time)
        data = data.dropna()
        for i in range(lag_order, len(data)):
            X.append(data.iloc[i - lag_order:i, :])
            Y.append(target.iloc[i - 1 + forecast_time])
        return np.array(X), np.array(Y)
