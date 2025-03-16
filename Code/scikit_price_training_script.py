import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from dataset import Dataset
from pipeline import Pipeline
from visualizations import Visualizer
from sklearn.linear_model import Ridge
import sklearn.preprocessing
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
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default


parser = argparse.ArgumentParser()
parser.add_argument("--ticker", type=str, choices=['btc', 'ltc', 'eth'], required=True,
                                        help="Cryptocurrency ticker (eth, ltc, or eth)")
args = parser.parse_args()
mlflow.set_experiment(args.ticker + "_16.3.2025")

# Filter out LinAlgWarning
warnings.filterwarnings("ignore", category=LinAlgWarning)
#controling whether tensorflow does recognize GPU
tf.config.get_visible_devices("GPU")
np.random.seed(42)
tf.random.set_seed(42)

mlflow.sklearn.autolog(disable=True)
pipeline = Pipeline(crypto_tick = args.ticker)
if args.ticker == "eth":
    pipeline.set_beginning(start_date = "2015-08-08")
else:
    pipeline.set_beginning(start_date = "2014-9-17")
    
pipeline.preprocess_dataset()
pipeline.data.drop(columns = [f'{args.ticker.upper()} / NVT, adjusted, free float,  90d MA',
                              f'{args.ticker.upper()} / NVT, adjusted, free float',
                              f'{args.ticker.upper()} / Fees, transaction, median, USD', 
                              f'{args.ticker.upper()} / Fees, total, USD',
                              f'{args.ticker.upper()} / Capitalization, market, free float, USD',
                              f'{args.ticker.upper()} / Capitalization, market, current supply, USD',
                              f'{args.ticker.upper()} / Capitalization, market, estimated supply, USD',
                              f'{args.ticker.upper()} / Difficulty, mean',
#                              f'{args.ticker.upper()} / Hash rate, mean, 30d',
                              f'{args.ticker.upper()} / Transactions, transfers, count',
                            f'{args.ticker.upper()} / Transactions, transfers, value, mean, USD'
                              ], inplace = True)

if args.ticker == "btc":
    pipeline.data.drop(columns = [f'{args.ticker.upper()} / Hash rate, mean, 30d'], inplace = True)



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
train_data_1, test_data_1, train_target_1, test_target_1 = Pipeline.split_train_test(pipeline.data_1d_shift.copy())
train_data_5, test_data_5, train_target_5, test_target_5 = Pipeline.split_train_test(pipeline.data_5d_shift.copy())
train_data_10, test_data_10, train_target_10, test_target_10 = Pipeline.split_train_test(pipeline.data_10d_shift.copy())
results_train_averaged["Naive forceast - 1 day"] = rmse(train_target_1, train_data_1.iloc[:,-1])
results_train_averaged["Naive forceast - 5 days"] = rmse(train_target_5, train_data_5.iloc[:,-1])
results_train_averaged["Naive forceast - 10 days"] = rmse(train_target_10, train_data_10.iloc[:,-1])
results_test["Naive forceast - 1 day"] = rmse(test_target_1, test_data_1.iloc[:,-1])
results_test["Naive forceast - 5 days"] = rmse(test_target_5, test_data_5.iloc[:,-1])
results_test["Naive forceast - 10 days"] = rmse(test_target_10, test_data_10.iloc[:,-1])

#Linear Regression
pipe = Pipeline.assembly_pipeline(estimator = Ridge(random_state = 42), dim_reducer = None)

LR_PARAMETERS = {"estimator__alpha": space.Real(0, 5000, prior = 'uniform'),
              "estimator__tol":space.Real(1e-5, 1, prior = 'log-uniform'),
              "estimator__max_iter":space.Integer(100, 10000000)}



#1 day
train_data, test_data, train_target, test_target = Pipeline.split_train_test(pipeline.data_1d_shift.copy())
model = Pipeline.fit_grid_search(train_data, train_target, test_data, test_target, pipe, LR_PARAMETERS, horizon = 1)
results_train_averaged.loc[["Full dimensionality"],[f"{args.ticker.upper()}-LR - 1 day"]] = rmse(train_target,
                                                                model.predict(train_data))

results_test.loc[["Full dimensionality"],[f"{args.ticker.upper()}-LR - 1 day"]] = rmse(test_target,
                                                                model.predict(test_data))
#5 days
train_data, test_data, train_target, test_target = Pipeline.split_train_test(pipeline.data_5d_shift.copy())
model = Pipeline.fit_grid_search(train_data, train_target, test_data, test_target, pipe, LR_PARAMETERS, horizon = 5)
results_train_averaged.loc[["Full dimensionality"],[f"{args.ticker.upper()}-LR - 5 days"]] = rmse(train_target,
                                                                model.predict(train_data))

results_test.loc[["Full dimensionality"],[f"{args.ticker.upper()}-LR - 5 days"]] = rmse(test_target,
                                                                model.predict(test_data))
#10 days
train_data, test_data, train_target, test_target = Pipeline.split_train_test(pipeline.data_10d_shift.copy())
model = Pipeline.fit_grid_search(train_data, train_target, test_data, test_target, pipe, LR_PARAMETERS, horizon = 10)
results_train_averaged.loc[["Full dimensionality"],[f"{args.ticker.upper()}-LR - 10 days"]] = rmse(train_target,
                                                                model.predict(train_data))

results_test.loc[["Full dimensionality"],[f"{args.ticker.upper()}-LR - 10 days"]] = rmse(test_target,
                                                                model.predict(test_data))

pca = PCA(n_components = 0.95)
pipe = Pipeline.assembly_pipeline(estimator = Ridge(random_state = 42), dim_reducer = pca)


#1 day
train_data, test_data, train_target, test_target = Pipeline.split_train_test(pipeline.data_1d_shift.copy())
model = Pipeline.fit_grid_search(train_data, train_target, test_data, test_target, pipe, LR_PARAMETERS, horizon = 1)
results_train_averaged.loc[["95% retained variance"],[f"{args.ticker.upper()}-LR - 1 day"]] = rmse(train_target,
                                                                model.predict(train_data))

results_test.loc[["95% retained variance"],[f"{args.ticker.upper()}-LR - 1 day"]] = rmse(test_target,
                                                                model.predict(test_data))
#5 days
train_data, test_data, train_target, test_target = Pipeline.split_train_test(pipeline.data_5d_shift.copy())
model = Pipeline.fit_grid_search(train_data, train_target, test_data, test_target, pipe, LR_PARAMETERS, horizon = 5)
results_train_averaged.loc[["95% retained variance"],[f"{args.ticker.upper()}-LR - 5 days"]] = rmse(train_target,
                                                                model.predict(train_data))

results_test.loc[["95% retained variance"],[f"{args.ticker.upper()}-LR - 5 days"]] = rmse(test_target,
                                                                model.predict(test_data))
#10 days
train_data, test_data, train_target, test_target = Pipeline.split_train_test(pipeline.data_10d_shift.copy())
model = Pipeline.fit_grid_search(train_data, train_target, test_data, test_target, pipe, LR_PARAMETERS, horizon = 10)
results_train_averaged.loc[["95% retained variance"],[f"{args.ticker.upper()}-LR - 10 days"]] = rmse(train_target,
                                                                model.predict(train_data))

results_test.loc[["95% retained variance"],[f"{args.ticker.upper()}-LR - 10 days"]] = rmse(test_target,
                                                                model.predict(test_data))

pca = PCA(n_components = 0.98)
pipe = Pipeline.assembly_pipeline(estimator = Ridge(random_state = 42), dim_reducer = pca)


#1 day
train_data, test_data, train_target, test_target = Pipeline.split_train_test(pipeline.data_1d_shift.copy())
model = Pipeline.fit_grid_search(train_data, train_target, test_data, test_target, pipe, LR_PARAMETERS, horizon = 1)
results_train_averaged.loc[["98% retained variance"],[f"{args.ticker.upper()}-LR - 1 day"]] = rmse(train_target,
                                                                model.predict(train_data))
prediction = model.predict(test_data)
results_test.loc[["98% retained variance"],[f"{args.ticker.upper()}-LR - 1 day"]] = rmse(test_target,
                                                                model.predict(test_data))
#5 days
train_data, test_data, train_target, test_target = Pipeline.split_train_test(pipeline.data_5d_shift.copy())
model = Pipeline.fit_grid_search(train_data, train_target, test_data, test_target, pipe, LR_PARAMETERS, horizon = 5)
results_train_averaged.loc[["98% retained variance"],[f"{args.ticker.upper()}-LR - 5 days"]] = rmse(train_target,
                                                                model.predict(train_data))
prediction = model.predict(test_data)
results_test.loc[["98% retained variance"],[f"{args.ticker.upper()}-LR - 5 days"]] = rmse(test_target,
                                                                model.predict(test_data))
#10 days
train_data, test_data, train_target, test_target = Pipeline.split_train_test(pipeline.data_10d_shift.copy())
model = Pipeline.fit_grid_search(train_data, train_target, test_data, test_target, pipe, LR_PARAMETERS, horizon = 10)
results_train_averaged.loc[["98% retained variance"],[f"{args.ticker.upper()}-LR - 10 days"]] = rmse(train_target,
                                                                model.predict(train_data))
prediction = model.predict(test_data)
results_test.loc[["98% retained variance"],[f"{args.ticker.upper()}-LR - 10 days"]] = rmse(test_target,
                                                                model.predict(test_data))

pca = PCA(n_components = 0.99)
pipe = Pipeline.assembly_pipeline(estimator = Ridge(random_state = 42), dim_reducer = pca)

#1 day
train_data, test_data, train_target, test_target = Pipeline.split_train_test(pipeline.data_1d_shift.copy())
model = Pipeline.fit_grid_search(train_data, train_target, test_data, test_target, pipe, LR_PARAMETERS, horizon = 1)
results_train_averaged.loc[["99% retained variance"],[f"{args.ticker.upper()}-LR - 1 day"]] = rmse(train_target,
                                                                model.predict(train_data))

results_test.loc[["99% retained variance"],[f"{args.ticker.upper()}-LR - 1 day"]] = rmse(test_target,
                                                                model.predict(test_data))
#5 days
train_data, test_data, train_target, test_target = Pipeline.split_train_test(pipeline.data_5d_shift.copy())
model = Pipeline.fit_grid_search(train_data, train_target, test_data, test_target, pipe, LR_PARAMETERS, horizon = 5)
results_train_averaged.loc[["99% retained variance"],[f"{args.ticker.upper()}-LR - 5 days"]] = rmse(train_target,
                                                                model.predict(train_data))

results_test.loc[["99% retained variance"],[f"{args.ticker.upper()}-LR - 5 days"]] = rmse(test_target,
                                                                model.predict(test_data))
#10 days
train_data, test_data, train_target, test_target = Pipeline.split_train_test(pipeline.data_10d_shift.copy())
model = Pipeline.fit_grid_search(train_data, train_target, test_data, test_target, pipe, LR_PARAMETERS, horizon = 10)
results_train_averaged.loc[["99% retained variance"],[f"{args.ticker.upper()}-LR - 10 days"]] = rmse(train_target,
                                                                model.predict(train_data))

results_test.loc[["99% retained variance"],[f"{args.ticker.upper()}-LR - 10 days"]] = rmse(test_target,
                                                                model.predict(test_data))


SVR_PARAMETERS = {"estimator__C": space.Real(1e-5, 10000, prior = 'uniform'),
              "estimator__tol":space.Real(1e-5, 1, prior = 'log-uniform'),
              "estimator__max_iter":space.Integer(100, 20000)}

#Support Vector Regression
pipe = Pipeline.assembly_pipeline(estimator = LinearSVR(random_state = 42,dual = "auto"), dim_reducer = None)

#1 day
train_data, test_data, train_target, test_target = Pipeline.split_train_test(pipeline.data_1d_shift.copy())
model = Pipeline.fit_grid_search(train_data, train_target, test_data, test_target, pipe, SVR_PARAMETERS, horizon = 1)
results_train_averaged.loc[["Full dimensionality"],[f"{args.ticker.upper()}-SVR - 1 day"]] = rmse(train_target,
                                                                model.predict(train_data))

results_test.loc[["Full dimensionality"],[f"{args.ticker.upper()}-SVR - 1 day"]] = rmse(test_target,
                                                                model.predict(test_data))
#5 days
train_data, test_data, train_target, test_target = Pipeline.split_train_test(pipeline.data_5d_shift.copy())
model = Pipeline.fit_grid_search(train_data, train_target, test_data, test_target, pipe, SVR_PARAMETERS, horizon = 5)
results_train_averaged.loc[["Full dimensionality"],[f"{args.ticker.upper()}-SVR - 5 days"]] = rmse(train_target,
                                                                model.predict(train_data))

results_test.loc[["Full dimensionality"],[f"{args.ticker.upper()}-SVR - 5 days"]] = rmse(test_target,
                                                                model.predict(test_data))
#10 days
train_data, test_data, train_target, test_target = Pipeline.split_train_test(pipeline.data_10d_shift.copy())
model = Pipeline.fit_grid_search(train_data, train_target, test_data, test_target, pipe, SVR_PARAMETERS, horizon = 10)
results_train_averaged.loc[["Full dimensionality"],[f"{args.ticker.upper()}-SVR - 10 days"]] = rmse(train_target,
                                                                model.predict(train_data))

results_test.loc[["Full dimensionality"],[f"{args.ticker.upper()}-SVR - 10 days"]] = rmse(test_target,
                                                                model.predict(test_data))

pca = PCA(n_components = 0.95)
pipe = Pipeline.assembly_pipeline(estimator = LinearSVR(random_state = 42), dim_reducer = pca)


#1 day
train_data, test_data, train_target, test_target = Pipeline.split_train_test(pipeline.data_1d_shift.copy())
model = Pipeline.fit_grid_search(train_data, train_target, test_data, test_target, pipe, SVR_PARAMETERS, horizon = 1)
results_train_averaged.loc[["95% retained variance"],[f"{args.ticker.upper()}-SVR - 1 day"]] = rmse(train_target,
                                                                model.predict(train_data))
prediction = model.predict(test_data)
results_test.loc[["95% retained variance"],[f"{args.ticker.upper()}-SVR - 1 day"]] = rmse(test_target,
                                                                prediction)
#5 days
train_data, test_data, train_target, test_target = Pipeline.split_train_test(pipeline.data_5d_shift.copy())
model = Pipeline.fit_grid_search(train_data, train_target, test_data, test_target, pipe, SVR_PARAMETERS, horizon = 5)
results_train_averaged.loc[["95% retained variance"],[f"{args.ticker.upper()}-SVR - 5 days"]] = rmse(train_target,
                                                                model.predict(train_data))
prediction = model.predict(test_data)
results_test.loc[["95% retained variance"],[f"{args.ticker.upper()}-SVR - 5 days"]] = rmse(test_target,
                                                                prediction)
#10 days
train_data, test_data, train_target, test_target = Pipeline.split_train_test(pipeline.data_10d_shift.copy())
model = Pipeline.fit_grid_search(train_data, train_target, test_data, test_target, pipe, SVR_PARAMETERS, horizon = 10)
results_train_averaged.loc[["95% retained variance"],[f"{args.ticker.upper()}-SVR - 10 days"]] = rmse(train_target,
                                                                model.predict(train_data))
prediction = model.predict(test_data)
results_test.loc[["95% retained variance"],[f"{args.ticker.upper()}-SVR - 10 days"]] = rmse(test_target,
                                                                prediction)

pca = PCA(n_components = 0.98)
pipe = Pipeline.assembly_pipeline(estimator = LinearSVR(random_state = 42), dim_reducer = pca)


#1 day
train_data, test_data, train_target, test_target = Pipeline.split_train_test(pipeline.data_1d_shift.copy())
model = Pipeline.fit_grid_search(train_data, train_target, test_data, test_target, pipe, SVR_PARAMETERS, horizon = 1)
results_train_averaged.loc[["98% retained variance"],[f"{args.ticker.upper()}-SVR - 1 day"]] = rmse(train_target,
                                                                model.predict(train_data))
prediction = model.predict(test_data)
results_test.loc[["98% retained variance"],[f"{args.ticker.upper()}-SVR - 1 day"]] = rmse(test_target,
                                                                prediction)
#5 days
train_data, test_data, train_target, test_target = Pipeline.split_train_test(pipeline.data_5d_shift.copy())
model = Pipeline.fit_grid_search(train_data, train_target, test_data, test_target, pipe, SVR_PARAMETERS, horizon = 5)
results_train_averaged.loc[["98% retained variance"],[f"{args.ticker.upper()}-SVR - 5 days"]] = rmse(train_target,
                                                                model.predict(train_data))
prediction = model.predict(test_data)
results_test.loc[["98% retained variance"],[f"{args.ticker.upper()}-SVR - 5 days"]] = rmse(test_target,
                                                                prediction)
#10 days
train_data, test_data, train_target, test_target = Pipeline.split_train_test(pipeline.data_10d_shift.copy())
model = Pipeline.fit_grid_search(train_data, train_target, test_data, test_target, pipe, SVR_PARAMETERS, horizon = 10)
results_train_averaged.loc[["98% retained variance"],[f"{args.ticker.upper()}-SVR - 10 days"]] = rmse(train_target,
                                                                model.predict(train_data))
prediction = model.predict(test_data)
results_test.loc[["98% retained variance"],[f"{args.ticker.upper()}-SVR - 10 days"]] = rmse(test_target,
                                                                prediction)

pca = PCA(n_components = 0.99)
pipe = Pipeline.assembly_pipeline(estimator = LinearSVR(random_state = 42), dim_reducer = pca)

#1 day
train_data, test_data, train_target, test_target = Pipeline.split_train_test(pipeline.data_1d_shift.copy())
model = Pipeline.fit_grid_search(train_data, train_target, test_data, test_target, pipe, SVR_PARAMETERS, horizon = 1)
results_train_averaged.loc[["99% retained variance"],[f"{args.ticker.upper()}-SVR - 1 day"]] = rmse(train_target,
                                                                model.predict(train_data))
prediction = model.predict(test_data)
results_test.loc[["99% retained variance"],[f"{args.ticker.upper()}-SVR - 1 day"]] = rmse(test_target,
                                                                prediction)
#5 days
train_data, test_data, train_target, test_target = Pipeline.split_train_test(pipeline.data_5d_shift.copy())
model = Pipeline.fit_grid_search(train_data, train_target, test_data, test_target, pipe, SVR_PARAMETERS, horizon = 5)
results_train_averaged.loc[["99% retained variance"],[f"{args.ticker.upper()}-SVR - 5 days"]] = rmse(train_target,
                                                                model.predict(train_data))
prediction = model.predict(test_data)
results_test.loc[["99% retained variance"],[f"{args.ticker.upper()}-SVR - 5 days"]] = rmse(test_target,
                                                                prediction)
#10 days
train_data, test_data, train_target, test_target = Pipeline.split_train_test(pipeline.data_10d_shift.copy())
model = Pipeline.fit_grid_search(train_data, train_target, test_data, test_target, pipe, SVR_PARAMETERS, horizon = 10)
results_train_averaged.loc[["99% retained variance"],[f"{args.ticker.upper()}-SVR - 10 days"]] = rmse(train_target,
                                                                model.predict(train_data))
prediction = model.predict(test_data)
results_test.loc[["99% retained variance"],[f"{args.ticker.upper()}-SVR - 10 days"]] = rmse(test_target,
                                                                prediction)

results_train_averaged.to_csv(f"results_train_averaged_{args.ticker.upper()}.csv")
results_test.to_csv(f"results_test_{args.ticker.upper()}.csv")
