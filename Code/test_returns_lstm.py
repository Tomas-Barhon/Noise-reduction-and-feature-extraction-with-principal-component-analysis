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
from sklearn.metrics import r2_score
import warnings
import mlflow
import argparse
from skopt import space, plots
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

print("GPUs:", tf.config.list_physical_devices('GPU'))
physical_devices = tf.config.list_physical_devices('GPU')
print(physical_devices)
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)


parser = argparse.ArgumentParser()
parser.add_argument("--ticker", type=str, choices=['btc', 'ltc', 'eth'], required=True,
                                        help="Cryptocurrency ticker (eth, ltc, or eth)")
args = parser.parse_args()
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

print(pipeline.data)

pipeline.shift_target()
columns = [f"{args.ticker.upper()}-LSTM - 1 day", 
                   f"{args.ticker.upper()}-LSTM - 5 days", f"{args.ticker.upper()}-LSTM - 10 days"]
rows = ["Full dimensionality", "95% retained variance",
                "98% retained variance", "99% retained variance"]
#presented in RMSE which is the optimized metric
results_train_averaged = pd.DataFrame(columns = columns, index = rows).fillna(0).astype(int)
columns = [f"{args.ticker.upper()}-LSTM - 1 day", f"{args.ticker.upper()}-LSTM - 5 days", 
                   f"{args.ticker.upper()}-LSTM - 10 days"]
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

print(pipeline.data_1d_shift)
x,y = Pipeline.create_lstm_input(pipeline.data_1d_shift.copy(), pipeline.data_1d_shift.copy().iloc[:,-1], 6)
train_data_1, test_data_1, train_target_1, test_target_1 = Pipeline.split_train_test([x,y], pandas=False)
index_1 = pipeline.data_1d_shift.index[6:]
split_point = int(len(index_1) * 0.8)
index_1_train = index_1[:split_point]
index_1_test = index_1[split_point:]
train_target_1 = cap_outliers(train_target_1)
x,y = Pipeline.create_lstm_input(pipeline.data_5d_shift.copy(), pipeline.data_5d_shift.copy().iloc[:,-1], 6)
train_data_5, test_data_5, train_target_5, test_target_5 = Pipeline.split_train_test([x,y], pandas=False)
index_5 = pipeline.data_5d_shift.index[6:]
split_point = int(len(index_5) * 0.8)
index_5_train = index_5[:split_point]
index_5_test = index_5[split_point:]
train_target_5 = cap_outliers(train_target_5)
x,y = Pipeline.create_lstm_input(pipeline.data_10d_shift.copy(), pipeline.data_10d_shift.copy().iloc[:,-1], 6)
train_data_10, test_data_10, train_target_10, test_target_10 = Pipeline.split_train_test([x,y], pandas=False)
index_10 = pipeline.data_10d_shift.index[6:]
split_point = int(len(index_10) * 0.8)
index_10_train = index_10[:split_point]
index_10_test = index_10[split_point:]
train_target_10 = cap_outliers(train_target_10)
#Linear Regression


LSTM_PARAMETERS = {"estimator__units": space.Integer(10, 300, prior = 'uniform'),
    "estimator__epochs": space.Integer(10, 200, prior = 'uniform'),
    "estimator__batch_size": space.Integer(5, 200, prior = 'uniform'),
    "estimator__dropout": space.Real(0, 0.5, prior = 'uniform'),
    "estimator__lr_initial": space.Real(1e-6, 1e-2, prior = 'log-uniform'),
    "estimator__layers": space.Categorical([1,2])}


#1 day
pipe = Pipeline.assembly_pipeline(
    estimator = LSTMRegressor(build_fn = Pipeline.assembly_lstm,
                    batch_size = 20,
                    epochs=150, 
                    input_shape=(6, len(pipeline.data_1d_shift.columns) -1),
                    units = 128,
                    dropout = 0.3,
                    lr_initial = 0.01,
                    layers = 2),
                    dim_reducer = None, 
                    shape_change = ((-1, len(pipeline.data_1d_shift.columns) -1), 
                                    (-1,6,len(pipeline.data_1d_shift.columns) -1)))
model = Pipeline.fit_grid_search(train_data_1, train_target_1, test_data_1, test_target_1, index_1_train, index_1_test, pipe, LSTM_PARAMETERS, n_jobs =None)
results_train_averaged.loc[["Full dimensionality"],[f"{args.ticker.upper()}-LSTM - 1 day"]] = round(r2_score(train_target_1,
                                                                model.predict(train_data_1)),5)

results_test.loc[["Full dimensionality"],[f"{args.ticker.upper()}-LSTM - 1 day"]] = round(r2_score(test_target_1,
                                                                model.predict(test_data_1)), 5)

pca = PCA(n_components = 0.95)
pipe = Pipeline.assembly_pipeline(
    estimator = LSTMRegressor(build_fn = Pipeline.assembly_lstm,
                    batch_size = 20,
                    epochs=150, 
                    input_shape=(6, len(pipeline.data_1d_shift.columns) -1),
                    units = 128,
                    dropout = 0.3,
                    lr_initial = 0.01,
                    layers = 2),
                    dim_reducer = pca, 
                    shape_change = ((-1, len(pipeline.data_1d_shift.columns) -1), 
                                    (-1,6,len(pipeline.data_1d_shift.columns) -1)))

model = Pipeline.fit_grid_search(train_data_1, train_target_1, test_data_1, test_target_1, index_1_train, index_1_test, pipe, LSTM_PARAMETERS, n_jobs =None)
results_train_averaged.loc[["95% retained variance"],[f"{args.ticker.upper()}-LSTM - 1 day"]] = round(r2_score(train_target_1,
                                                                model.predict(train_data_1)),5)

results_test.loc[["95% retained variance"],[f"{args.ticker.upper()}-LSTM - 1 day"]] = round(r2_score(test_target_1,
                                                                model.predict(test_data_1)), 5)


pca = PCA(n_components = 0.98)
pipe = Pipeline.assembly_pipeline(
    estimator = LSTMRegressor(build_fn = Pipeline.assembly_lstm,
                    batch_size = 20,
                    epochs=150, 
                    input_shape=(6, len(pipeline.data_1d_shift.columns) -1),
                    units = 128,
                    dropout = 0.3,
                    lr_initial = 0.01,
                    layers = 2),
                    dim_reducer = pca, 
                    shape_change = ((-1, len(pipeline.data_1d_shift.columns) -1), 
                                    (-1,6,len(pipeline.data_1d_shift.columns) -1)))
model = Pipeline.fit_grid_search(train_data_1, train_target_1, test_data_1, test_target_1, index_1_train, index_1_test, pipe, LSTM_PARAMETERS, n_jobs =None)
results_train_averaged.loc[["98% retained variance"],[f"{args.ticker.upper()}-LSTM - 1 day"]] = round(r2_score(train_target_1,
                                                                model.predict(train_data_1)),5)

results_test.loc[["98% retained variance"],[f"{args.ticker.upper()}-LSTM - 1 day"]] = round(r2_score(test_target_1,
                                                                model.predict(test_data_1)), 5)
pca = PCA(n_components = 0.99)
pipe = Pipeline.assembly_pipeline(
    estimator = LSTMRegressor(build_fn = Pipeline.assembly_lstm,
                    batch_size = 20,
                    epochs=150, 
                    input_shape=(6, len(pipeline.data_1d_shift.columns) -1),
                    units = 128,
                    dropout = 0.3,
                    lr_initial = 0.01,
                    layers = 2),
                    dim_reducer = pca, 
                    shape_change = ((-1, len(pipeline.data_1d_shift.columns) -1), 
                                    (-1,6,len(pipeline.data_1d_shift.columns) -1)))

model = Pipeline.fit_grid_search(train_data_1, train_target_1, test_data_1, test_target_1, index_1_train, index_1_test, pipe, LSTM_PARAMETERS, n_jobs =None)
results_train_averaged.loc[["99% retained variance"],[f"{args.ticker.upper()}-LSTM - 1 day"]] = round(r2_score(train_target_1,
                                                                model.predict(train_data_1)),5)

results_test.loc[["99% retained variance"],[f"{args.ticker.upper()}-LSTM - 1 day"]] = round(r2_score(test_target_1,
                                                                model.predict(test_data_1)), 5)

#5 days
pipe = Pipeline.assembly_pipeline(
    estimator = LSTMRegressor(build_fn = Pipeline.assembly_lstm,
                    batch_size = 20,
                    epochs=150, 
                    input_shape=(6, len(pipeline.data_1d_shift.columns) -1),
                    units = 128,
                    dropout = 0.3,
                    lr_initial = 0.01,
                    layers = 2),
                    dim_reducer = None, 
                    shape_change = ((-1, len(pipeline.data_1d_shift.columns) -1), 
                                    (-1,6,len(pipeline.data_1d_shift.columns) -1)))
model = Pipeline.fit_grid_search(train_data_5, train_target_5, test_data_5, test_target_5, index_5_train, index_5_test, pipe, LSTM_PARAMETERS, n_jobs =None)
results_train_averaged.loc[["Full dimensionality"],[f"{args.ticker.upper()}-LSTM - 1 day"]] = round(r2_score(train_target_5,
                                                                model.predict(train_data_5)),5)

results_test.loc[["Full dimensionality"],[f"{args.ticker.upper()}-LSTM - 1 day"]] = round(r2_score(test_target_5,
                                                                model.predict(test_data_1)), 5)

pca = PCA(n_components = 0.95)
pipe = Pipeline.assembly_pipeline(
    estimator = LSTMRegressor(build_fn = Pipeline.assembly_lstm,
                    batch_size = 20,
                    epochs=150, 
                    input_shape=(6, len(pipeline.data_1d_shift.columns) -1),
                    units = 128,
                    dropout = 0.3,
                    lr_initial = 0.01,
                    layers = 2),
                    dim_reducer = pca, 
                    shape_change = ((-1, len(pipeline.data_1d_shift.columns) -1), 
                                    (-1,6,len(pipeline.data_1d_shift.columns) -1)))

model = Pipeline.fit_grid_search(train_data_5, train_target_5, test_data_5, test_target_5, index_5_train, index_5_test, pipe, LSTM_PARAMETERS, n_jobs =None)
results_train_averaged.loc[["95% retained variance"],[f"{args.ticker.upper()}-LSTM - 1 day"]] = round(r2_score(train_target_5,
                                                                model.predict(train_data_5)),5)

results_test.loc[["95% retained variance"],[f"{args.ticker.upper()}-LSTM - 1 day"]] = round(r2_score(test_target_5,
                                                                model.predict(test_data_1)), 5)


pca = PCA(n_components = 0.98)
pipe = Pipeline.assembly_pipeline(
    estimator = LSTMRegressor(build_fn = Pipeline.assembly_lstm,
                    batch_size = 20,
                    epochs=150, 
                    input_shape=(6, len(pipeline.data_1d_shift.columns) -1),
                    units = 128,
                    dropout = 0.3,
                    lr_initial = 0.01,
                    layers = 2),
                    dim_reducer = pca, 
                    shape_change = ((-1, len(pipeline.data_1d_shift.columns) -1), 
                                    (-1,6,len(pipeline.data_1d_shift.columns) -1)))
model = Pipeline.fit_grid_search(train_data_5, train_target_5, test_data_5, test_target_5, index_5_train, index_5_test, pipe, LSTM_PARAMETERS, n_jobs =None)
results_train_averaged.loc[["98% retained variance"],[f"{args.ticker.upper()}-LSTM - 1 day"]] = round(r2_score(train_target_5,
                                                                model.predict(train_data_5)),5)

results_test.loc[["98% retained variance"],[f"{args.ticker.upper()}-LSTM - 1 day"]] = round(r2_score(test_target_5,
                                                                model.predict(test_data_1)), 5)
pca = PCA(n_components = 0.99)
pipe = Pipeline.assembly_pipeline(
    estimator = LSTMRegressor(build_fn = Pipeline.assembly_lstm,
                    batch_size = 20,
                    epochs=150, 
                    input_shape=(6, len(pipeline.data_1d_shift.columns) -1),
                    units = 128,
                    dropout = 0.3,
                    lr_initial = 0.01,
                    layers = 2),
                    dim_reducer = pca, 
                    shape_change = ((-1, len(pipeline.data_1d_shift.columns) -1), 
                                    (-1,6,len(pipeline.data_1d_shift.columns) -1)))

model = Pipeline.fit_grid_search(train_data_5, train_target_5, test_data_5, test_target_5, index_5_train, index_5_test, pipe, LSTM_PARAMETERS, n_jobs =None)
results_train_averaged.loc[["99% retained variance"],[f"{args.ticker.upper()}-LSTM - 1 day"]] = round(r2_score(train_target_5,
                                                                model.predict(train_data_5)),5)

results_test.loc[["99% retained variance"],[f"{args.ticker.upper()}-LSTM - 1 day"]] = round(r2_score(test_target_5,
                                                                model.predict(test_data_1)), 5)

#10 days
pipe = Pipeline.assembly_pipeline(
    estimator = LSTMRegressor(build_fn = Pipeline.assembly_lstm,
                    batch_size = 20,
                    epochs=150, 
                    input_shape=(6, len(pipeline.data_1d_shift.columns) -1),
                    units = 128,
                    dropout = 0.3,
                    lr_initial = 0.01,
                    layers = 2),
                    dim_reducer = None, 
                    shape_change = ((-1, len(pipeline.data_1d_shift.columns) -1), 
                                    (-1,6,len(pipeline.data_1d_shift.columns) -1)))
model = Pipeline.fit_grid_search(train_data_10, train_target_10, test_data_10, test_target_10, index_10_train, index_10_test, pipe, LSTM_PARAMETERS, n_jobs =None)
results_train_averaged.loc[["Full dimensionality"],[f"{args.ticker.upper()}-LSTM - 1 day"]] = round(r2_score(train_target_10,
                                                                model.predict(train_data_10)),5)

results_test.loc[["Full dimensionality"],[f"{args.ticker.upper()}-LSTM - 1 day"]] = round(r2_score(test_target_10,
                                                                model.predict(test_data_10)), 5)

pca = PCA(n_components = 0.95)
pipe = Pipeline.assembly_pipeline(
    estimator = LSTMRegressor(build_fn = Pipeline.assembly_lstm,
                    batch_size = 20,
                    epochs=150, 
                    input_shape=(6, len(pipeline.data_1d_shift.columns) -1),
                    units = 128,
                    dropout = 0.3,
                    lr_initial = 0.01,
                    layers = 2),
                    dim_reducer = pca, 
                    shape_change = ((-1, len(pipeline.data_1d_shift.columns) -1), 
                                    (-1,6,len(pipeline.data_1d_shift.columns) -1)))

model = Pipeline.fit_grid_search(train_data_10, train_target_10, test_data_10, test_target_10, index_10_train, index_10_test, pipe, LSTM_PARAMETERS, n_jobs =None)
results_train_averaged.loc[["95% retained variance"],[f"{args.ticker.upper()}-LSTM - 1 day"]] = round(r2_score(train_target_10,
                                                                model.predict(train_data_10)),5)

results_test.loc[["95% retained variance"],[f"{args.ticker.upper()}-LSTM - 1 day"]] = round(r2_score(test_target_10,
                                                                model.predict(test_data_10)), 5)


pca = PCA(n_components = 0.98)
pipe = Pipeline.assembly_pipeline(
    estimator = LSTMRegressor(build_fn = Pipeline.assembly_lstm,
                    batch_size = 20,
                    epochs=150, 
                    input_shape=(6, len(pipeline.data_1d_shift.columns) -1),
                    units = 128,
                    dropout = 0.3,
                    lr_initial = 0.01,
                    layers = 2),
                    dim_reducer = pca, 
                    shape_change = ((-1, len(pipeline.data_1d_shift.columns) -1), 
                                    (-1,6,len(pipeline.data_1d_shift.columns) -1)))
model = Pipeline.fit_grid_search(train_data_10, train_target_10, test_data_10, test_target_10, index_10_train, index_10_test, pipe, LSTM_PARAMETERS, n_jobs =None)
results_train_averaged.loc[["98% retained variance"],[f"{args.ticker.upper()}-LSTM - 1 day"]] = round(r2_score(train_target_10,
                                                                model.predict(train_data_10)),5)

results_test.loc[["98% retained variance"],[f"{args.ticker.upper()}-LSTM - 1 day"]] = round(r2_score(test_target_10,
                                                                model.predict(test_data_10)), 5)
pca = PCA(n_components = 0.99)
pipe = Pipeline.assembly_pipeline(
    estimator = LSTMRegressor(build_fn = Pipeline.assembly_lstm,
                    batch_size = 20,
                    epochs=150, 
                    input_shape=(6, len(pipeline.data_1d_shift.columns) -1),
                    units = 128,
                    dropout = 0.3,
                    lr_initial = 0.01,
                    layers = 2),
                    dim_reducer = pca, 
                    shape_change = ((-1, len(pipeline.data_1d_shift.columns) -1), 
                                    (-1,6,len(pipeline.data_1d_shift.columns) -1)))

model = Pipeline.fit_grid_search(train_data_10, train_target_10, test_data_10, test_target_10, index_10_train, index_10_test, pipe, LSTM_PARAMETERS, n_jobs =None)
results_train_averaged.loc[["99% retained variance"],[f"{args.ticker.upper()}-LSTM - 1 day"]] = round(r2_score(train_target_10,
                                                                model.predict(train_data_10)),5)

results_test.loc[["99% retained variance"],[f"{args.ticker.upper()}-LSTM - 1 day"]] = round(r2_score(test_target_10,
                                                                model.predict(test_data_10)), 5)
results_train_averaged.to_csv(f"results_train_averaged_{args.ticker.upper()}.csv")
results_test.to_csv(f"results_test_{args.ticker.upper()}.csv")