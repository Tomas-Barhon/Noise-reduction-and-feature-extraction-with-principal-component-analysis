import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default
import tensorflow as tf
import numpy as np
import pandas as pd
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
# Filter out LinAlgWarning
warnings.filterwarnings("ignore", category=LinAlgWarning)
#controling whether tensorflow does recognize GPU
tf.config.get_visible_devices("GPU")
np.random.seed(42)
tf.random.set_seed(42)

mlflow.autolog()
pipeline = Pipeline(crypto_tick = "btc")
pipeline.set_beginning(start_date = "2014-9-17")
pipeline.preprocess_dataset()
pipeline.data.drop(columns = ['BTC / NVT, adjusted, free float,  90d MA',
                              'BTC / NVT, adjusted, free float',
                              'BTC / Fees, transaction, median, USD', 'BTC / Fees, total, USD',
                              'BTC / Capitalization, market, free float, USD',
                              'BTC / Capitalization, market, current supply, USD',
                              'BTC / Capitalization, market, estimated supply, USD',
                              'BTC / Difficulty, mean',
                              'BTC / Hash rate, mean, 30d',
                              'BTC / Transactions, transfers, count',
                            'BTC / Transactions, transfers, value, mean, USD'
                              ], inplace = True)
pipeline.shift_target()
columns = ["BTC-LR - 1 day", "BTC-LR - 5 days", 
           "BTC-LR - 10 days", "BTC-SVR - 1 day", "BTC-SVR - 5 days", 
           "BTC-SVR - 10 days", "BTC-LSTM - 1 day", "BTC-LSTM - 5 days","BTC-LSTM - 10 days", "Naive forceast - 1 day", "Naive forceast - 5 days",
           "Naive forceast - 10 days"]
rows = ["Full dimensionality", "95% retained variance",
        "98% retained variance", "99% retained variance"]
#presented in RMSE which is the optimized metric
results_train_averaged = pd.DataFrame(columns = columns, index = rows).fillna(0).astype(int)
columns = ["BTC-LR - 1 day", "BTC-LR - 5 days", 
           "BTC-LR - 10 days", "BTC-SVR - 1 day", "BTC-SVR - 5 days", "BTC-SVR - 10 days", 
           "BTC-LSTM - 1 day", "BTC-LSTM - 5 days","BTC-LSTM - 10 days", "Naive forceast - 1 day", "Naive forceast - 5 days",
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

LR_PARAMETERS = {"estimator__alpha": np.linspace(0,7,20),
              "estimator__tol":[0.0001, 0.0005,0.001],
              "estimator__max_iter":[200,500,1000,2000,5000,10000]}

#1 day
train_data, test_data, train_target, test_target = Pipeline.split_train_test(pipeline.data_1d_shift.copy())
model = Pipeline.fit_grid_search(train_data, train_target, pipe, LR_PARAMETERS)
results_train_averaged.loc[["Full dimensionality"],["BTC-LR - 1 day"]] = rmse(train_target,
                                                                model.predict(train_data))
prediction = model.predict(test_data)
results_test.loc[["Full dimensionality"],["BTC-LR - 1 day"]] = rmse(test_target,
                                                                prediction)
#5 days
train_data, test_data, train_target, test_target = Pipeline.split_train_test(pipeline.data_5d_shift.copy())
model = Pipeline.fit_grid_search(train_data, train_target, pipe, LR_PARAMETERS)
results_train_averaged.loc[["Full dimensionality"],["BTC-LR - 5 day"]] = rmse(train_target,
                                                                model.predict(train_data))
prediction = model.predict(test_data)
results_test.loc[["Full dimensionality"],["BTC-LR - 5 day"]] = rmse(test_target,
                                                                prediction)
#10 days
train_data, test_data, train_target, test_target = Pipeline.split_train_test(pipeline.data_10d_shift.copy())
model = Pipeline.fit_grid_search(train_data, train_target, pipe, LR_PARAMETERS)
results_train_averaged.loc[["Full dimensionality"],["BTC-LR - 10 day"]] = rmse(train_target,
                                                                model.predict(train_data))
prediction = model.predict(test_data)
results_test.loc[["Full dimensionality"],["BTC-LR - 10 day"]] = rmse(test_target,
                                                                prediction)

pca = PCA(n_components = 0.95)
pipe = Pipeline.assembly_pipeline(estimator = Ridge(random_state = 42), dim_reducer = pca)


#1 day
train_data, test_data, train_target, test_target = Pipeline.split_train_test(pipeline.data_1d_shift.copy())
model = Pipeline.fit_grid_search(train_data, train_target, pipe, LR_PARAMETERS)
results_train_averaged.loc[["Full dimensionality"],["BTC-LR - 1 day"]] = rmse(train_target,
                                                                model.predict(train_data))
prediction = model.predict(test_data)
results_test.loc[["Full dimensionality"],["BTC-LR - 1 day"]] = rmse(test_target,
                                                                prediction)
#5 days
train_data, test_data, train_target, test_target = Pipeline.split_train_test(pipeline.data_5d_shift.copy())
model = Pipeline.fit_grid_search(train_data, train_target, pipe, LR_PARAMETERS)
results_train_averaged.loc[["Full dimensionality"],["BTC-LR - 5 day"]] = rmse(train_target,
                                                                model.predict(train_data))
prediction = model.predict(test_data)
results_test.loc[["Full dimensionality"],["BTC-LR - 5 day"]] = rmse(test_target,
                                                                prediction)
#10 days
train_data, test_data, train_target, test_target = Pipeline.split_train_test(pipeline.data_10d_shift.copy())
model = Pipeline.fit_grid_search(train_data, train_target, pipe, LR_PARAMETERS)
results_train_averaged.loc[["Full dimensionality"],["BTC-LR - 10 day"]] = rmse(train_target,
                                                                model.predict(train_data))
prediction = model.predict(test_data)
results_test.loc[["Full dimensionality"],["BTC-LR - 10 day"]] = rmse(test_target,
                                                                prediction)

pca = PCA(n_components = 0.98)
pipe = Pipeline.assembly_pipeline(estimator = Ridge(random_state = 42), dim_reducer = pca)


#1 day
train_data, test_data, train_target, test_target = Pipeline.split_train_test(pipeline.data_1d_shift.copy())
model = Pipeline.fit_grid_search(train_data, train_target, pipe, LR_PARAMETERS)
results_train_averaged.loc[["Full dimensionality"],["BTC-LR - 1 day"]] = rmse(train_target,
                                                                model.predict(train_data))
prediction = model.predict(test_data)
results_test.loc[["Full dimensionality"],["BTC-LR - 1 day"]] = rmse(test_target,
                                                                prediction)
#5 days
train_data, test_data, train_target, test_target = Pipeline.split_train_test(pipeline.data_5d_shift.copy())
model = Pipeline.fit_grid_search(train_data, train_target, pipe, LR_PARAMETERS)
results_train_averaged.loc[["Full dimensionality"],["BTC-LR - 5 day"]] = rmse(train_target,
                                                                model.predict(train_data))
prediction = model.predict(test_data)
results_test.loc[["Full dimensionality"],["BTC-LR - 5 day"]] = rmse(test_target,
                                                                prediction)
#10 days
train_data, test_data, train_target, test_target = Pipeline.split_train_test(pipeline.data_10d_shift.copy())
model = Pipeline.fit_grid_search(train_data, train_target, pipe, LR_PARAMETERS)
results_train_averaged.loc[["Full dimensionality"],["BTC-LR - 10 day"]] = rmse(train_target,
                                                                model.predict(train_data))
prediction = model.predict(test_data)
results_test.loc[["Full dimensionality"],["BTC-LR - 10 day"]] = rmse(test_target,
                                                                prediction)

pca = PCA(n_components = 0.99)
pipe = Pipeline.assembly_pipeline(estimator = Ridge(random_state = 42), dim_reducer = pca)

#1 day
train_data, test_data, train_target, test_target = Pipeline.split_train_test(pipeline.data_1d_shift.copy())
model = Pipeline.fit_grid_search(train_data, train_target, pipe, LR_PARAMETERS)
results_train_averaged.loc[["Full dimensionality"],["BTC-LR - 1 day"]] = rmse(train_target,
                                                                model.predict(train_data))
prediction = model.predict(test_data)
results_test.loc[["Full dimensionality"],["BTC-LR - 1 day"]] = rmse(test_target,
                                                                prediction)
#5 days
train_data, test_data, train_target, test_target = Pipeline.split_train_test(pipeline.data_5d_shift.copy())
model = Pipeline.fit_grid_search(train_data, train_target, pipe, LR_PARAMETERS)
results_train_averaged.loc[["Full dimensionality"],["BTC-LR - 5 day"]] = rmse(train_target,
                                                                model.predict(train_data))
prediction = model.predict(test_data)
results_test.loc[["Full dimensionality"],["BTC-LR - 5 day"]] = rmse(test_target,
                                                                prediction)
#10 days
train_data, test_data, train_target, test_target = Pipeline.split_train_test(pipeline.data_10d_shift.copy())
model = Pipeline.fit_grid_search(train_data, train_target, pipe, LR_PARAMETERS)
results_train_averaged.loc[["Full dimensionality"],["BTC-LR - 10 day"]] = rmse(train_target,
                                                                model.predict(train_data))
prediction = model.predict(test_data)
results_test.loc[["Full dimensionality"],["BTC-LR - 10 day"]] = rmse(test_target,
                                                                prediction)


SVR_PARAMETERS = {"estimator__C": np.logspace(1,10,20),
              "estimator__tol":[0.00005,0.0001, 0.0005,0.001],
              "estimator__max_iter":[5000,10000,20000]}
#Support Vector Regression
pipe = Pipeline.assembly_pipeline(estimator = LinearSVR(random_state = 42,dual = "auto"), dim_reducer = None)

#1 day
train_data, test_data, train_target, test_target = Pipeline.split_train_test(pipeline.data_1d_shift.copy())
model = Pipeline.fit_grid_search(train_data, train_target, pipe, SVR_PARAMETERS)
results_train_averaged.loc[["Full dimensionality"],["BTC-LR - 1 day"]] = rmse(train_target,
                                                                model.predict(train_data))
prediction = model.predict(test_data)
results_test.loc[["Full dimensionality"],["BTC-LR - 1 day"]] = rmse(test_target,
                                                                prediction)
#5 days
train_data, test_data, train_target, test_target = Pipeline.split_train_test(pipeline.data_5d_shift.copy())
model = Pipeline.fit_grid_search(train_data, train_target, pipe, SVR_PARAMETERS)
results_train_averaged.loc[["Full dimensionality"],["BTC-LR - 5 day"]] = rmse(train_target,
                                                                model.predict(train_data))
prediction = model.predict(test_data)
results_test.loc[["Full dimensionality"],["BTC-LR - 5 day"]] = rmse(test_target,
                                                                prediction)
#10 days
train_data, test_data, train_target, test_target = Pipeline.split_train_test(pipeline.data_10d_shift.copy())
model = Pipeline.fit_grid_search(train_data, train_target, pipe, SVR_PARAMETERS)
results_train_averaged.loc[["Full dimensionality"],["BTC-LR - 10 day"]] = rmse(train_target,
                                                                model.predict(train_data))
prediction = model.predict(test_data)
results_test.loc[["Full dimensionality"],["BTC-LR - 10 day"]] = rmse(test_target,
                                                                prediction)

pca = PCA(n_components = 0.95)
pipe = Pipeline.assembly_pipeline(estimator = LinearSVR(random_state = 42), dim_reducer = pca)


#1 day
train_data, test_data, train_target, test_target = Pipeline.split_train_test(pipeline.data_1d_shift.copy())
model = Pipeline.fit_grid_search(train_data, train_target, pipe, SVR_PARAMETERS)
results_train_averaged.loc[["Full dimensionality"],["BTC-LR - 1 day"]] = rmse(train_target,
                                                                model.predict(train_data))
prediction = model.predict(test_data)
results_test.loc[["Full dimensionality"],["BTC-LR - 1 day"]] = rmse(test_target,
                                                                prediction)
#5 days
train_data, test_data, train_target, test_target = Pipeline.split_train_test(pipeline.data_5d_shift.copy())
model = Pipeline.fit_grid_search(train_data, train_target, pipe, SVR_PARAMETERS)
results_train_averaged.loc[["Full dimensionality"],["BTC-LR - 5 day"]] = rmse(train_target,
                                                                model.predict(train_data))
prediction = model.predict(test_data)
results_test.loc[["Full dimensionality"],["BTC-LR - 5 day"]] = rmse(test_target,
                                                                prediction)
#10 days
train_data, test_data, train_target, test_target = Pipeline.split_train_test(pipeline.data_10d_shift.copy())
model = Pipeline.fit_grid_search(train_data, train_target, pipe, SVR_PARAMETERS)
results_train_averaged.loc[["Full dimensionality"],["BTC-LR - 10 day"]] = rmse(train_target,
                                                                model.predict(train_data))
prediction = model.predict(test_data)
results_test.loc[["Full dimensionality"],["BTC-LR - 10 day"]] = rmse(test_target,
                                                                prediction)

pca = PCA(n_components = 0.98)
pipe = Pipeline.assembly_pipeline(estimator = LinearSVR(random_state = 42), dim_reducer = pca)


#1 day
train_data, test_data, train_target, test_target = Pipeline.split_train_test(pipeline.data_1d_shift.copy())
model = Pipeline.fit_grid_search(train_data, train_target, pipe, SVR_PARAMETERS)
results_train_averaged.loc[["Full dimensionality"],["BTC-LR - 1 day"]] = rmse(train_target,
                                                                model.predict(train_data))
prediction = model.predict(test_data)
results_test.loc[["Full dimensionality"],["BTC-LR - 1 day"]] = rmse(test_target,
                                                                prediction)
#5 days
train_data, test_data, train_target, test_target = Pipeline.split_train_test(pipeline.data_5d_shift.copy())
model = Pipeline.fit_grid_search(train_data, train_target, pipe, SVR_PARAMETERS)
results_train_averaged.loc[["Full dimensionality"],["BTC-LR - 5 day"]] = rmse(train_target,
                                                                model.predict(train_data))
prediction = model.predict(test_data)
results_test.loc[["Full dimensionality"],["BTC-LR - 5 day"]] = rmse(test_target,
                                                                prediction)
#10 days
train_data, test_data, train_target, test_target = Pipeline.split_train_test(pipeline.data_10d_shift.copy())
model = Pipeline.fit_grid_search(train_data, train_target, pipe, SVR_PARAMETERS)
results_train_averaged.loc[["Full dimensionality"],["BTC-LR - 10 day"]] = rmse(train_target,
                                                                model.predict(train_data))
prediction = model.predict(test_data)
results_test.loc[["Full dimensionality"],["BTC-LR - 10 day"]] = rmse(test_target,
                                                                prediction)

pca = PCA(n_components = 0.99)
pipe = Pipeline.assembly_pipeline(estimator = LinearSVR(random_state = 42), dim_reducer = pca)

#1 day
train_data, test_data, train_target, test_target = Pipeline.split_train_test(pipeline.data_1d_shift.copy())
model = Pipeline.fit_grid_search(train_data, train_target, pipe, SVR_PARAMETERS)
results_train_averaged.loc[["Full dimensionality"],["BTC-LR - 1 day"]] = rmse(train_target,
                                                                model.predict(train_data))
prediction = model.predict(test_data)
results_test.loc[["Full dimensionality"],["BTC-LR - 1 day"]] = rmse(test_target,
                                                                prediction)
#5 days
train_data, test_data, train_target, test_target = Pipeline.split_train_test(pipeline.data_5d_shift.copy())
model = Pipeline.fit_grid_search(train_data, train_target, pipe, SVR_PARAMETERS)
results_train_averaged.loc[["Full dimensionality"],["BTC-LR - 5 day"]] = rmse(train_target,
                                                                model.predict(train_data))
prediction = model.predict(test_data)
results_test.loc[["Full dimensionality"],["BTC-LR - 5 day"]] = rmse(test_target,
                                                                prediction)
#10 days
train_data, test_data, train_target, test_target = Pipeline.split_train_test(pipeline.data_10d_shift.copy())
model = Pipeline.fit_grid_search(train_data, train_target, pipe, SVR_PARAMETERS)
results_train_averaged.loc[["Full dimensionality"],["BTC-LR - 10 day"]] = rmse(train_target,
                                                                model.predict(train_data))
prediction = model.predict(test_data)
results_test.loc[["Full dimensionality"],["BTC-LR - 10 day"]] = rmse(test_target,
                                                                prediction)