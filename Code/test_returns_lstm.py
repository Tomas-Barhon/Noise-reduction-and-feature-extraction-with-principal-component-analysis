import os
import numpy as np
import pandas as pd
import tensorflow as tf
#print("GPUs:", tf.config.list_physical_devices('GPU'))
import matplotlib.pyplot as plt
import seaborn as sns
from dataset import Dataset
from pipeline import Pipeline, LSTMRegressor
from visualizations import Visualizer
from sklearn.linear_model import Ridge, HuberRegressor
import sklearn.preprocessing
from sklearn.preprocessing import MinMaxScaler
import statsmodels.api as sm
from sklearn.svm import LinearSVR, SVR
from sklearn.decomposition import PCA, KernelPCA
from scikeras.wrappers import KerasRegressor
from sklearn.metrics import mean_squared_error
from sktime.performance_metrics.forecasting import mean_squared_percentage_error
from sktime.performance_metrics.forecasting import MeanSquaredError
rmse = MeanSquaredError(square_root = True)
from scipy.linalg import LinAlgWarning
import sklearn.model_selection
import warnings
import mlflow
import argparse
from skopt import space, plots
#os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

print("GPUs:", tf.config.list_physical_devices('GPU'))
physical_devices = tf.config.list_physical_devices('GPU')
print(physical_devices)
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)


class Args():
    def __init__(self):
        self.ticker = "eth"
args = Args()
args.ticker = "eth"
mlflow.set_experiment(args.ticker + "_5.4.2025-returns")

# Filter out LinAlgWarning
warnings.filterwarnings("ignore", category=LinAlgWarning)
#controling whether tensorflow does recognize GPU
tf.config.get_visible_devices("GPU")
np.random.seed(42)
tf.random.set_seed(42)

mlflow.sklearn.autolog(disable=True)
pipeline = Pipeline(crypto_tick = args.ticker, returns=True)
if args.ticker == "eth":
    pipeline.set_beginning(start_date = "2015-08-08")
else:
    pipeline.set_beginning(start_date = "2014-9-17")
     
pipeline.preprocess_dataset()
print(pipeline.data.info())
pipeline.data.iloc[:,:-1] = np.log(pipeline.data.iloc[:, :-1]).diff()

pipeline.data.drop(columns = ['Close_GC=F', 'Close_^IXIC', 'Close_VGT', 'Close_XSD', 'Close_IYW', 'Close_FTEC', 'Close_IGV',
       'RGDP_US', 'RGDP_PC_US', 'M2_US'], inplace = True)

pipeline.shift_target()
columns = [f"{args.ticker.upper()}-LR - 1 day", f"{args.ticker.upper()}-LR - 5 days", 
                   f"{args.ticker.upper()}-LR - 10 days", f"{args.ticker.upper()}-SVR - 1 day", 
                   f"{args.ticker.upper()}-SVR - 5 days", 
                   f"{args.ticker.upper()}-SVR - 10 days", f"{args.ticker.upper()}-LSTM - 1 day", 
                   f"{args.ticker.upper()}-LSTM - 5 days", f"{args.ticker.upper()}-LSTM - 10 days", 
                   "Naive forceast - 1 day", "Naive forceast - 5 days",
                   "Naive forceast - 10 days"]
rows = ["Full dimensionality", "95% retained variance",
                "98% retained variance", "99% retained variance"]
#presented in RMSE which is the optimized metric
results_train_averaged = pd.DataFrame(columns = columns, index = rows).fillna(0).astype(int)
columns = [f"{args.ticker.upper()}-LR - 1 day", f"{args.ticker.upper()}-LR - 5 days", 
                   f"{args.ticker.upper()}-LR - 10 days", 
                   f"{args.ticker.upper()}-SVR - 1 day", f"{args.ticker.upper()}-SVR - 5 days", 
                   f"{args.ticker.upper()}-SVR - 10 days", 
                   f"{args.ticker.upper()}-LSTM - 1 day", f"{args.ticker.upper()}-LSTM - 5 days", 
                   f"{args.ticker.upper()}-LSTM - 10 days", "Naive forceast - 1 day", "Naive forceast - 5 days",
                   "Naive forceast - 10 days"]
rows = ["Full dimensionality", "95% retained variance",
                "98% retained variance", "99% retained variance"]
#presented in RMSE which is the optimized metric
results_test = pd.DataFrame(columns = columns, index = rows).fillna(0).astype(int)


#Naive forecast
# Cap outliers on the train_target datasets
def cap_outliers(data, lower_percentile=2, upper_percentile=98):
    lower_bound = np.percentile(data, lower_percentile)
    upper_bound = np.percentile(data, upper_percentile)
    return np.clip(data, lower_bound, upper_bound)


x,y = Pipeline.create_lstm_input(pipeline.data_1d_shift.copy(), pipeline.data_1d_shift.copy().iloc[:,-1], 6)
train_data_1, test_data_1, train_target_1, test_target_1 = Pipeline.split_train_test([x,y], pandas=False)
index_1 = pipeline.data_1d_shift.index[6:]
split_point = int(len(index_1) * 0.8)
index_1_train = index_1[:split_point]
index_1_test = index_1[split_point:]
#train_target_1 = cap_outliers(train_target_1)
x,y = Pipeline.create_lstm_input(pipeline.data_5d_shift.copy(), pipeline.data_5d_shift.copy().iloc[:,-1], 6)
train_data_5, test_data_5, train_target_5, test_target_5 = Pipeline.split_train_test([x,y], pandas=False)
index_5 = pipeline.data_5d_shift.index[6:]
split_point = int(len(index_5) * 0.8)
index_5_train = index_5[:split_point]
index_5_test = index_5[split_point:]
#train_target_5 = cap_outliers(train_target_5)
x,y = Pipeline.create_lstm_input(pipeline.data_10d_shift.copy(), pipeline.data_10d_shift.copy().iloc[:,-1], 6)
train_data_10, test_data_10, train_target_10, test_target_10 = Pipeline.split_train_test([x,y], pandas=False)
index_10 = pipeline.data_10d_shift.index[6:]
split_point = int(len(index_10) * 0.8)
index_10_train = index_10[:split_point]
index_10_test = index_10[split_point:]
#train_target_10 = cap_outliers(train_target_10)
results_train_averaged["Naive forceast - 1 day"] = rmse(train_target_1, np.zeros_like(train_target_1))
results_train_averaged["Naive forceast - 5 days"] = rmse(train_target_5, np.zeros_like(train_target_5))
results_train_averaged["Naive forceast - 10 days"] = rmse(train_target_10, np.zeros_like(train_target_10))
results_test["Naive forceast - 1 day"] = rmse(test_target_1, np.zeros_like(test_target_1))
results_test["Naive forceast - 5 days"] = rmse(test_target_5, np.zeros_like(test_target_5))
results_test["Naive forceast - 10 days"] = rmse(test_target_10, np.zeros_like(test_target_10))

print(rmse(train_target_1, np.zeros_like(train_target_1)))
print(rmse(train_target_5, np.zeros_like(train_target_5)))
print(rmse(train_target_10, np.zeros_like(train_target_10)))
print(rmse(test_target_1, np.zeros_like(test_target_1)))
print(rmse(test_target_5, np.zeros_like(test_target_5)))
print(rmse(test_target_10, np.zeros_like(test_target_10)))
#Linear Regression
pca = PCA(n_components = 0.99)
pipe = Pipeline.assembly_pipeline(
    estimator = LSTMRegressor(build_fn = Pipeline.assembly_lstm,
                    batch_size = 5,
                    epochs=150, 
                    input_shape=(6, len(pipeline.data_1d_shift.columns) -1),
                    units = 64,
                    dropout = 0.5,
                    lr_initial = 0.001,
                    layers = 2),
                    dim_reducer = None, 
                    shape_change = ((-1, len(pipeline.data_1d_shift.columns) -1), 
                                    (-1,6,len(pipeline.data_1d_shift.columns) -1)))


LSTM_PARAMETERS = {"estimator__units": space.Integer(10, 200, prior = 'uniform'),
    "estimator__epochs": space.Integer(10, 200, prior = 'uniform'),
    "estimator__batch_size": space.Integer(5, 200, prior = 'uniform'),
    "estimator__dropout": space.Real(0, 0.5, prior = 'uniform'),
    "estimator__lr_initial": space.Real(1e-6, 1e-2, prior = 'log-uniform'),
    "estimator__layers": space.Categorical([1,2])}

print(test_data_10.shape)
# minimax = sklearn.preprocessing.MinMaxScaler((0,1))
# train_target_1 = np.squeeze(minimax.fit_transform(train_target_1.reshape(-1, 1)))
# test_target_1 = np.squeeze(minimax.transform(test_target_1.reshape(-1, 1)))
def plot_acf_comparison(train_series, test_series, lags=50):
    """
    Plots the autocorrelation function (ACF) for train and test datasets.
    
    Parameters:
    - train_series: np.array or pd.Series, training returns series
    - test_series: np.array or pd.Series, test returns series
    - lags: int, number of lags to plot
    """
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # ACF for training data
    sm.graphics.tsa.plot_acf(train_series, lags=lags, ax=axes[0])
    axes[0].set_title("Train Data - ACF")
    
    # ACF for test data
    sm.graphics.tsa.plot_acf(test_series, lags=lags, ax=axes[1])
    axes[1].set_title("Test Data - ACF")
    
    plt.tight_layout()
    plt.show()

# Example Usage (replace with actual series)
# Assuming train_data and test_data are 1D NumPy arrays or Pandas Series
#plot_acf_comparison(train_target_1, test_target_1, lags=50)
# model = Pipeline.fit_grid_search(train_data_10, train_target_10, test_data_10, test_target_10, index_10_train, index_10_test, pipe, LSTM_PARAMETERS, n_jobs =None)
# print(pipe.named_steps)
#minimax = sklearn.preprocessing.MinMaxScaler((0,1))
#train_target = np.squeeze(minimax.fit_transform(train_target_10.reshape(-1, 1)))
#test_target = np.squeeze(minimax.transform(test_target_10.reshape(-1, 1)))

pipe.fit(train_data_10, train_target_10)



plt.plot(np.concatenate([pipe.predict(train_data_10), pipe.predict(test_data_10)]), linewidth=0.2)
plt.plot(np.concatenate([train_target_10,test_target_10]), linewidth=0.2)
plt.savefig("returns_lstm.png")
print(rmse(train_target_10, pipe.predict(train_data_10)))
print(rmse(test_target_10, pipe.predict(test_data_10)))

