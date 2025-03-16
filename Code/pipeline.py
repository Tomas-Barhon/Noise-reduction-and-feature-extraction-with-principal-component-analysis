from typing import Literal
from dataset import Dataset
import sklearn.model_selection
import sklearn.preprocessing
import sklearn.pipeline
import tensorflow as tf
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from skopt import BayesSearchCV
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
        """Shifts the target variable by 1,5 and 10 days back ensuring that 
        we use historical explanatory variables to predict future target variable
        and dropping the nonoverlapping entries.

        Returns:
            _type_: _description_
        """
        # 1 day forecasting
        if self.returns:
            self.data_1d_shift = self.data.copy()
            self.data_1d_shift.iloc[:, -1] =  np.log(self.data.iloc[:, -1].shift(1)).diff()
            self.data_1d_shift["target"] = np.log(self.data.iloc[:, -1].shift(-1)).diff()
            self.data_1d_shift = self.data_1d_shift.dropna()
            
            # 5 day forecasting
            self.data_5d_shift = self.data.copy()
            self.data_5d_shift.iloc[:, -1] =  np.log(self.data.iloc[:, -1].shift(1)).diff()
            self.data_5d_shift["target"] = np.log(self.data.iloc[:, -1].shift(-5)).diff()
            self.data_5d_shift = self.data_5d_shift.dropna()
            
            # 10 day forecasting
            self.data_10d_shift = self.data.copy()
            self.data_10d_shift.iloc[:, -1] =  np.log(self.data.iloc[:, -1].shift(1)).diff()
            self.data_10d_shift["target"] = np.log(self.data.iloc[:, -1].shift(-10)).diff()
            self.data_10d_shift = self.data_10d_shift.dropna()
        else:
            self.data_1d_shift = self.data.copy()
            self.data_1d_shift["target"] = self.data.iloc[:, -1].shift(-1)
            self.data_1d_shift = self.data_1d_shift.dropna()
            
            # 5 day forecasting
            self.data_5d_shift = self.data.copy()
            self.data_5d_shift["target"] = self.data.iloc[:, -1].shift(-5)
            self.data_5d_shift = self.data_5d_shift.dropna()
            
            # 10 day forecasting
            self.data_10d_shift = self.data.copy()
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
                data.iloc[:, :-1], data.iloc[:, -1], test_size=0.1, random_state=42, shuffle=False)
        else:
            train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
                data[0], data[1], test_size=0.1, random_state=42, shuffle=False)
        return train_data, test_data, train_target, test_target

    @staticmethod
    def assembly_pipeline(estimator, dim_reducer, shape_change=False):
        """
        Creates sklearn.pipeline with specified steps. 
        Takes care of shape transformations for LSTM.
        """
        scaler = sklearn.preprocessing.StandardScaler()
        denoiser = None
        scaler_2 = None
        unpacker = None
        packer = None
        if dim_reducer is not None:
            denoiser = PCATransformer(dim_reducer)
            scaler_2 = sklearn.preprocessing.StandardScaler()

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
    def fit_grid_search(train_data, train_target, test_data, test_target, pipeline, parameter_grid, n_jobs=-1, horizon = 1):
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
        scoring = {"RMSE": "neg_root_mean_squared_error",
                   "MAE": "neg_mean_absolute_error",
                   "MAPE": "neg_mean_absolute_percentage_error"}
        ts_split = sklearn.model_selection.TimeSeriesSplit(n_splits=2)
        model = BayesSearchCV(
            pipeline, search_spaces=parameter_grid,
            cv=3, scoring=scoring, refit="RMSE", n_points=4,
            verbose=1, n_jobs=n_jobs, error_score='raise').fit(train_data, train_target)

        estimator_name = type(model.best_estimator_.named_steps["estimator"]).__name__
        n_components = ""
        if model.best_estimator_.named_steps["denoiser"] is not None:
            n_components = "_pca_" + model.best_estimator_.named_steps["denoiser"].pca.n_components_
        with mlflow.start_run(run_name=estimator_name + str(n_components) + "_h-" + str(horizon)):
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
            test_prediction = pd.Series(test_prediction, index=test_data.index)
            train_pred = pd.Series(y_pred, index=train_data.index)
            visualizer = Visualizer()
            fig = visualizer.draw_prediction_full(train_target,train_pred, test_target, test_prediction, horizon)
            mlflow.log_figure(fig, "prediction_plot_full.html")
            fig = visualizer.draw_prediction_test(test_target, test_prediction, horizon)
            mlflow.log_figure(fig, "prediction_plot_test.html")
        return model


    @staticmethod
    def fit_full_train_grid_search(train_data, train_target, pipeline, parameter_grid):
        ...
        

    @staticmethod
    def assembly_lstm(input_shape, units, dropout, lr_initial):
        """
        Creates LSTM network with layer normalization for stable training
        """
        model = tf.keras.Sequential()
        
        # First LSTM block with layer normalization
        model.add(tf.keras.layers.LayerNormalization(input_shape=input_shape))
        model.add(tf.keras.layers.LSTM(units, return_sequences=True))
        model.add(tf.keras.layers.Dropout(dropout))
        
        # Second LSTM block with layer normalization
        model.add(tf.keras.layers.LayerNormalization())
        model.add(tf.keras.layers.LSTM(units))
        model.add(tf.keras.layers.Dropout(dropout))
        
        # Dense layers
        model.add(tf.keras.layers.Dense(1))
        
        # Define learning rate schedule with exponential decay
        initial_learning_rate = lr_initial
        decay_steps = 1500
        decay_rate = 0.9
        learning_rate_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate,
            decay_steps=decay_steps,
            decay_rate=decay_rate,
            staircase=True
        )

        model.compile(
            optimizer=tf.keras.optimizers.AdamW(learning_rate=learning_rate_schedule),
            loss=Pipeline.root_mean_squared_error
        )
        
        # Create TensorBoard callback
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir='./logs',
            histogram_freq=1,
            write_graph=True,
            write_images=True
        )
        # Store callbacks as a model property to be used during fit
        model.callbacks_list = [tensorboard_callback]
        
        return model

    @staticmethod
    def root_mean_squared_error(y_true, y_pred):
        """Custom loss function for tensorflow RMSE.

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            RMSE loss value
        """
        return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true)))

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
