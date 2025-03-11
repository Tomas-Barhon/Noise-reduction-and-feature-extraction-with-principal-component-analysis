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
#os.chdir("./Bachelor thesis/Code")
# Filter out LinAlgWarning
warnings.filterwarnings("ignore", category=LinAlgWarning)
#controling whether tensorflow does recognize GPU
tf.config.get_visible_devices("GPU")
np.random.seed(42)
tf.random.set_seed(42)

LSTM_PARAMETERS = {"estimator__units": [700,800,900],
    "estimator__epochs": [1000],
    "estimator__batch_size": [100],
    "estimator__dropout": [0.1,0.2,0.5],
    "estimator__lr_initial": [0.001,0.01,0.1],}


#Inititate processing pipeline
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

#Naive Forecasting
train_data_1, test_data_1, train_target_1, test_target_1 = Pipeline.split_train_test(pipeline.data_1d_shift.copy())
train_data_5, test_data_5, train_target_5, test_target_5 = Pipeline.split_train_test(pipeline.data_5d_shift.copy())
train_data_10, test_data_10, train_target_10, test_target_10 = Pipeline.split_train_test(pipeline.data_10d_shift.copy())
results_train_averaged["Naive forceast - 1 day"] = rmse(train_target_1, train_data_1.iloc[:,-1])
results_train_averaged["Naive forceast - 5 days"] = rmse(train_target_5, train_data_5.iloc[:,-1])
results_train_averaged["Naive forceast - 10 days"] = rmse(train_target_10, train_data_10.iloc[:,-1])
results_test["Naive forceast - 1 day"] = rmse(test_target_1, test_data_1.iloc[:,-1])
results_test["Naive forceast - 5 days"] = rmse(test_target_5, test_data_5.iloc[:,-1])
results_test["Naive forceast - 10 days"] = rmse(test_target_10, test_data_10.iloc[:,-1])

#1 day
pipe = Pipeline.assembly_pipeline(estimator = KerasRegressor(model = Pipeline.assembly_lstm,
                    verbose=1, random_state = 42, shuffle = True, 
                    batch_size = 200,epochs=300, input_shape=(3, 41),
                    units = 500, dropout = 0.2,lr_initial = 0.01), dim_reducer = None, shape_change = ((-1, 41), (-1,3,41)))
x,y = Pipeline.create_lstm_input(pipeline.data_1d_shift.copy(), pipeline.data_1d_shift.copy().iloc[:,-1], 3)
train_data, test_data, train_target, test_target = Pipeline.split_train_test([x,y], pandas = False)
model = Pipeline.fit_grid_search(train_data, train_target, pipe, LSTM_PARAMETERS, n_jobs =None)
results_train_averaged.loc[["Full dimensionality"],["BTC-LSTM - 1 day"]] = rmse(train_target.reshape(-1,1), model.predict(train_data).reshape(-1,1))
prediction = model.predict(test_data)
results_test.loc[["Full dimensionality"],["BTC-LSTM - 1 day"]] = rmse(test_target.reshape(-1,1), 
                                                                      model.predict(test_data).reshape(-1,1))

#5 day
x,y = Pipeline.create_lstm_input(pipeline.data_5d_shift.copy(), pipeline.data_5d_shift.copy().iloc[:,-1], 3)
train_data, test_data, train_target, test_target = Pipeline.split_train_test([x,y], pandas = False)
model = Pipeline.fit_grid_search(train_data, train_target, pipe, LSTM_PARAMETERS, n_jobs =None)
results_train_averaged.loc[["Full dimensionality"],["BTC-LSTM - 5 day"]] = rmse(train_target.reshape(-1,1), model.predict(train_data).reshape(-1,1))
prediction = model.predict(test_data)
results_test.loc[["Full dimensionality"],["BTC-LSTM - 5 day"]] = rmse(test_target.reshape(-1,1), 
                                                                      model.predict(test_data).reshape(-1,1))
#10 day
x,y = Pipeline.create_lstm_input(pipeline.data_10d_shift.copy(), pipeline.data_10d_shift.copy().iloc[:,-1], 3)
train_data, test_data, train_target, test_target = Pipeline.split_train_test([x,y], pandas = False)
model = Pipeline.fit_grid_search(train_data, train_target, pipe, LSTM_PARAMETERS, n_jobs =None)
results_train_averaged.loc[["Full dimensionality"],["BTC-LSTM - 10 day"]] = rmse(train_target.reshape(-1,1), model.predict(train_data).reshape(-1,1))
prediction = model.predict(test_data)
results_test.loc[["Full dimensionality"],["BTC-LSTM - 10 day"]] = rmse(test_target.reshape(-1,1), 
                                                                      model.predict(test_data).reshape(-1,1))