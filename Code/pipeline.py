import sys
from typing import Literal
from dataset import Dataset
import sklearn.model_selection
from sklearn.model_selection import KFold
import sklearn.preprocessing
import sklearn.pipeline
import tensorflow as tf
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from tensorflow.keras.callbacks import TensorBoard
import numpy as np
from skopt import BayesSearchCV
from sklearn.utils.validation import check_X_y, check_array
import mlflow
import mlflow.sklearn
from visualizations import Visualizer
import pandas as pd 

class PCATransformer(BaseEstimator, TransformerMixin):
    """_summary_

    Pushes original data into n dimensions based on retained percantage 
    of variance specified in pca. And upsamples them back into the original 
    number of dimensions.

    Args:
        BaseEstimator (_type_): _description_
        TransformerMixin (_type_): _description_
    """
    def __init__(self, pca):
        self.pca = pca

    def fit(self, X, y=None):
        """
        Fit the PCA model on the provided data.
        Parameters:
        X : array-like, shape (n_samples, n_features)
            Training data.
        y : Ignored
            Not used, present here for API consistency by convention.
        Returns:
        self : object
            Returns the instance itself.
        """
    
        self.pca.fit(X)
        return self

    def transform(self, X):
        """
        Transforms the input data by applying PCA downsampling and then restoring it to the original dimensions.
        Parameters:
        X (array-like): The input data to be transformed.
        Returns:
        array-like: The transformed data, restored to the original dimensions.
        """
        
        # Downsample to n_components
        X_pca = self.pca.transform(X)
        # Upsample back to original dimensions
        X_restored = self.pca.inverse_transform(X_pca)
        return X_pca

class LSTMRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, build_fn, input_shape, units=32, dropout=0.3,
                 lr_initial=0.001, layers=1, epochs=1, batch_size=32, verbose=10):
        if build_fn is None:
            raise ValueError("build_fn cannot be None. It must be a function that returns a compiled Keras model.")

        self.build_fn = build_fn
        self.input_shape = input_shape
        self.units = units
        self.dropout = dropout
        self.lr_initial = lr_initial
        self.layers = layers
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.model_ = None

    def fit(self, X, y):
        #X, y = check_X_y(X, y, ensure_2d=False)  # Validate input
        self.model_ = self.build_fn(
            input_shape=self.input_shape, 
            units=self.units, 
            dropout=self.dropout, 
            lr_initial=self.lr_initial, 
            layers=self.layers,
            epochs = self.epochs
        )
        tensorboard_cb = TensorBoard(log_dir="./logs", histogram_freq=1, write_graph=True)
        self.model_.fit(X, y, epochs=self.epochs, batch_size=self.batch_size,
                        callbacks=[tensorboard_cb])
        sys.stdout.flush()
        return self

    def predict(self, X):
        X = np.asarray(X)  # Ensure it's a NumPy array
        if X.ndim != 3:
            raise ValueError(f"Expected X to be 3D (samples, timesteps, features), but got shape {X.shape}.")
        return self.model_.predict(X)

    def score(self, X, y):
        X, y = check_X_y(X, y, ensure_2d=False)
        y_pred = self.predict(X)
        return -np.sqrt(np.mean((y - y_pred) ** 2))  # Negative RMSE

    def get_params(self, deep=True):
        return {
            "build_fn": self.build_fn, 
            "input_shape": self.input_shape, 
            "units": self.units, 
            "dropout": self.dropout, 
            "lr_initial": self.lr_initial, 
            "layers": self.layers, 
            "epochs": self.epochs, 
            "batch_size": self.batch_size, 
            "verbose": self.verbose
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self


class ReshapeTransformer(BaseEstimator, TransformerMixin):
    """_summary_

    Reshaping layer for sklearn.pipeline.

    Args:
        BaseEstimator (_type_): _description_
        TransformerMixin (_type_): _description_
    """
    def __init__(self, new_shape):
        self.new_shape = new_shape

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.reshape(self.new_shape)


class Pipeline:
    """
    
    Pipeline class takes which takes the ticker of either ("btc", "ltc", "eth")
    and creates all shifted versions of the dataset we use in our thesis.
    It represents the whole processing workflow that we performed on the data.
    """
    def __init__(self, crypto_tick: Literal["btc", "ltc", "eth"], returns = False) -> None:
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
        self.returns = returns

    def set_beginning(self, start_date: str):
        self.data = self.data.loc[start_date:]
        return self

    def preprocess_dataset(self):
        """
        Fills missing data with zeros where the data was not yet collected and 
        forward fills all the exchange data that are not traded on the weekends 
        (pushing values from friday to saturday and sunday mostly).
        Cuts ethereum time series before at the point of switching to P-o-s consensus.
        
        """
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
#                self.data["BTC / Capitalization, market, estimated supply, USD"] = self.data[
#                    "BTC / Capitalization, market, estimated supply, USD"].fillna(0)
                self.data = self.data.loc[:"2024-01-20"]
            case "ltc":
                self.data['Wiki_ltc_search'] = self.data['Wiki_ltc_search'].fillna(
                    0)
#                self.data["LTC / Capitalization, market, estimated supply, USD"] = self.data[
#                    "LTC / Capitalization, market, estimated supply, USD"].fillna(0)
            case "eth":
                self.data['Wiki_eth_search'] = self.data['Wiki_eth_search'].fillna(
                    0)
                self.data["ETH / Mean Hash Rate"] = self.data["ETH / Mean Hash Rate"].ffill()
                self.data["ETH / Miner Revenue per Hash (USD)"] = self.data[
                    "ETH / Miner Revenue per Hash (USD)"].ffill()
#                self.data["ETH / Capitalization, market, estimated supply, USD"] = self.data[
#                    "ETH / Capitalization, market, estimated supply, USD"].fillna(0)
                # cutting the end where the PoS consensus mechanism starts
                self.data = self.data.loc[:"2023-12-21"]
                # dropping the beginning where the moving averages are not present
                self.data = self.data.dropna()
        return self

    def shift_target(self):
        """Shifts the target variable by 1,5 and 10 days back ensuring that 
        we use historical explanatory variables to predict future target variable
        and dropping the nonoverlapping entries.

        Returns:
            _type_: _description_
        """

        if self.returns:
            def sma(data, window=14):
                return data.rolling(window=window).mean()

            # Calculate Exponential Moving Average (EMA)
            def ema(data, window=14):
                return data.ewm(span=window, adjust=False).mean()

            # Calculate Relative Strength Index (RSI)
            def rsi(data, window=14):
                delta = data.diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
                rs = gain / loss
                return 100 - (100 / (1 + rs))

            # Calculate Bollinger Bands
            def bollinger_bands(data, window=14, num_std=2):
                sma_value = sma(data, window)
                rolling_std = data.rolling(window=window).std()
                upper_band = sma_value + (rolling_std * num_std)
                lower_band = sma_value - (rolling_std * num_std)
                return upper_band, lower_band

            # Add technical indicators to the data
            def add_technical_indicators(df):
                # Adding SMA, EMA, RSI, and Bollinger Bands
                df['sma_14'] = sma(df['returns_today'], window=14)
                df['ema_14'] = ema(df['returns_today'], window=14)
                df['rsi_14'] = rsi(df['returns_today'], window=14)
                df['bb_upper'], df['bb_lower'] = bollinger_bands(df['returns_today'], window=14)
                return df

            self.data_1d_shift = self.data.copy()
            self.data_1d_shift["returns_today"] = np.log(self.data.iloc[:, -1] / self.data.iloc[:, -1].shift(1))
            self.data_1d_shift = self.data_1d_shift.dropna()
            self.data_1d_shift = add_technical_indicators(self.data_1d_shift)
            self.data_1d_shift["target"] = self.data_1d_shift["returns_today"].shift(-1)
            self.data_1d_shift = self.data_1d_shift.dropna()
            # 5-day forecasting
            self.data_5d_shift = self.data.copy()
            self.data_5d_shift["returns_today"] = np.log(self.data.iloc[:, -1] / self.data.iloc[:, -1].shift(5))
            self.data_5d_shift = self.data_5d_shift.dropna()
            self.data_5d_shift = add_technical_indicators(self.data_5d_shift)
            self.data_5d_shift["target"] = self.data_5d_shift["returns_today"].shift(-5)
            self.data_5d_shift = self.data_5d_shift.dropna()
            # 10-day forecasting
            self.data_10d_shift = self.data.copy()
            self.data_10d_shift["returns_today"] = np.log(self.data.iloc[:, -1] / self.data.iloc[:, -1].shift(10))
            self.data_10d_shift = self.data_10d_shift.dropna()
            self.data_10d_shift = add_technical_indicators(self.data_10d_shift)
            self.data_10d_shift["target"] = self.data_10d_shift["returns_today"].shift(-10)
            self.data_10d_shift = self.data_10d_shift.dropna()
        else:
            self.data_1d_shift = self.data.copy()
            # Create lagged values
            lag_1 = self.data.iloc[:, -1].shift(1)
            lag_2 = self.data.iloc[:, -1].shift(2)
            # Insert lags before the last column
            self.data_1d_shift.insert(len(self.data_1d_shift.columns)-1, 'lag_1', lag_1)
            self.data_1d_shift.insert(len(self.data_1d_shift.columns)-1, 'lag_2', lag_2)
            self.data_1d_shift["target"] = self.data.iloc[:, -1].shift(-1)
            self.data_1d_shift = self.data_1d_shift.dropna()
            
            # 5 day forecasting
            self.data_5d_shift = self.data.copy()
            self.data_5d_shift.insert(len(self.data_5d_shift.columns)-1, 'lag_1', lag_1)
            self.data_5d_shift.insert(len(self.data_5d_shift.columns)-1, 'lag_2', lag_2)
            self.data_5d_shift["target"] = self.data.iloc[:, -1].shift(-5)
            self.data_5d_shift = self.data_5d_shift.dropna()
            
            # 10 day forecasting
            self.data_10d_shift = self.data.copy()
            self.data_10d_shift.insert(len(self.data_10d_shift.columns)-1, 'lag_1', lag_1)
            self.data_10d_shift.insert(len(self.data_10d_shift.columns)-1, 'lag_2', lag_2)
            self.data_10d_shift["target"] = self.data.iloc[:, -1].shift(-10)
            self.data_10d_shift = self.data_10d_shift.dropna()
        return self
        



    @staticmethod
    def split_train_test(data, pandas=True):
        """_summary_

        Splits the data into first 90% training and last 10% testing without shuffling them to 
        ensure keeping the time series characteristic of the problem.
        Args:
            data (_type_): _description_
            pandas (bool, optional): _description_. Defaults to True.

        Returns:
            _type_: _description_
        """
        if pandas:
            train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
                data.iloc[:, :-1], data.iloc[:, -1], test_size=0.2, random_state=42, shuffle=False)
        else:
            train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
                data[0], data[1], test_size=0.2, random_state=42, shuffle=False)
            
        return train_data, test_data, train_target, test_target

    @staticmethod
    def assembly_pipeline(estimator, dim_reducer, shape_change=False):
        """
        Creates sklearn.pipeline with specified steps. 
        Takes care of shape transformations for LSTM.
        """
        if shape_change is not False:
            scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(-1,1))
        else:
            scaler = sklearn.preprocessing.RobustScaler(unit_variance=True)
        denoiser = None
        scaler_2 = None
        unpacker = None
        packer = None
        if dim_reducer is not None:
            denoiser = PCATransformer(dim_reducer)
            #scaler_2 = sklearn.preprocessing.StandardScaler()

        if shape_change is not False:
            unpacker = ReshapeTransformer(shape_change[0])
            packer = ReshapeTransformer(shape_change[1])
        pipeline = sklearn.pipeline.Pipeline([("pack_down", unpacker),
                                              ("scaler", scaler),
                                              ("denoiser", denoiser),
            #                                 ("scaler 2", scaler_2),
                                              ("pack_up", packer),
                                              ("estimator", estimator)])

        return pipeline

    @staticmethod
    def fit_grid_search(train_data, train_target, test_data, test_target, train_index, test_index, pipeline, parameter_grid, n_jobs=-1, horizon = 1):
        """Fits grid search using the time series split with all metrics using
        the best RMSE as the best model.

        Args:
            train_data (_type_): _description_
            train_target (_type_): _description_
            pipeline (_type_): _description_
            parameter_grid (_type_): _description_
            n_jobs (int, optional): _description_. Defaults to 5.

        Returns:
            _type_: _description_
        """
        scoring = {"RMSE": "neg_root_mean_squared_error"}
        ts_split = sklearn.model_selection.TimeSeriesSplit(n_splits=3)

        model = BayesSearchCV(
            pipeline, search_spaces=parameter_grid,
            cv=ts_split, scoring=scoring, refit="RMSE", n_points=4,
            verbose=3, n_jobs=n_jobs, error_score='raise', n_iter=20).fit(train_data, train_target)

        estimator_name = type(model.best_estimator_.named_steps["estimator"]).__name__
        n_components = ""
        if model.best_estimator_.named_steps["denoiser"] is not None:
            n_components = "_pca_" + str(model.best_estimator_.named_steps["denoiser"].pca.n_components_)
        with mlflow.start_run(run_name=estimator_name + n_components + "_h-" + str(horizon)):
            # Log cross validation results
            for metric in scoring.keys():
                cv_results_mean = model.cv_results_[f'mean_test_{metric}']
                mlflow.log_metric(f"cv_mean_{metric}", -cv_results_mean[model.best_index_])
                
            mlflow.sklearn.log_model(
            model.best_estimator_,
            "best_model",
            registered_model_name=estimator_name
            )
            mlflow.log_params(model.best_params_)
            test_prediction = model.predict(test_data)
            # Log metrics from the refit model
            y_pred = model.predict(train_data)
            rmse = np.sqrt(sklearn.metrics.mean_squared_error(train_target, y_pred))
            mlflow.log_metric("RMSE_train", rmse)
            rmse_test = np.sqrt(sklearn.metrics.mean_squared_error(test_target, test_prediction))
            mlflow.log_metric("RMSE_test", rmse_test)
            print(test_prediction.shape)
            test_prediction = pd.Series(np.squeeze(test_prediction), index=test_index)
            train_pred = pd.Series(np.squeeze(y_pred), index=train_index)
            train_target = pd.Series(np.squeeze(train_target), index=train_index)
            test_target = pd.Series(np.squeeze(test_target), index=test_index)
            visualizer = Visualizer()
            fig = visualizer.draw_prediction_full(train_target,train_pred, test_target, test_prediction, horizon)
            mlflow.log_figure(fig, "prediction_plot_full.png")
            fig = visualizer.draw_prediction_test(test_target, test_prediction, horizon)
            mlflow.log_figure(fig, "prediction_plot_test.png")
        return model


    @staticmethod
    def fit_full_train_grid_search(train_data, train_target, pipeline, parameter_grid):
        ...
        

    @staticmethod
    def assembly_lstm(input_shape, units, dropout, lr_initial, layers, epochs):
        """
        Creates LSTM network with layer normalization for stable training
        """
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.InputLayer(input_shape=input_shape))
        # First LSTM block
        #model.add(tf.keras.layers.BatchNormalization())
        if layers == 2:
            model.add(tf.keras.layers.LSTM(units, return_sequences=True))
        else:
            model.add(tf.keras.layers.LSTM(units, return_sequences=False))
        model.add(tf.keras.layers.LayerNormalization())
        model.add(tf.keras.layers.Dropout
                    (dropout))
        # Second LSTM block
        if layers == 2:
            model.add(tf.keras.layers.LSTM(units//2))
            model.add(tf.keras.layers.LayerNormalization())
            model.add(tf.keras.layers.Dropout
                    (dropout))
        
        # Dense layers
        model.add(tf.keras.layers.Dense(units//2, activation="relu"))
        model.add(tf.keras.layers.Dropout
        (dropout))
        model.add(tf.keras.layers.Dense(1))
        
        # Define learning rate schedule with exponential decay
        initial_learning_rate = lr_initial
        decay_steps = epochs
        decay_rate = 0.9
        learning_rate_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate,
            decay_steps=decay_steps,
            decay_rate=decay_rate,
            staircase=True
        )

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate_schedule),
            loss="mse", metrics = [Pipeline.root_mean_squared_error], steps_per_execution=10
        )
        print(model.summary())
        return model

    @staticmethod
    def root_mean_squared_error(y_true, y_pred):
        mse = tf.keras.losses.MeanSquaredError()(y_true, y_pred)  # Instantiate and call
        return tf.keras.backend.sqrt(mse)

    @staticmethod
    def create_lstm_input(data, target, lag_order, forecast_time = 1):
        """
        Creates the input for the LSTM network. Predicting target at time t 
        with variables from t-lag_order - t-1.
        """
        X, Y = [], []
        for i in range(lag_order, len(data)):
            X.append(data.iloc[i - lag_order:i, :-1])
            Y.append(target.iloc[i])
        return np.array(X), np.array(Y)
