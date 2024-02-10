import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns


class Visualizer():
    TARGET_COLOR = "black"
    PREDICTION_COLOR = "tab:blue"
    STYLE = "seaborn-v0_8-notebook"

    def __init__(self) -> None:
        plt.style.use(self.STYLE)

    @staticmethod
    def draw_prediction_test(test_target, test_prediction):
        fig, ax = plt.subplots(1, 1)
        ax.plot(test_target, label="Target", color=Visualizer.TARGET_COLOR)
        ax.plot(test_prediction, label="Prediction",
                color=Visualizer.PREDICTION_COLOR)
        ax.tick_params(axis='x', labelrotation=90)
        ax.set(xlabel='Date', ylabel='Price in USD$')
        ax.yaxis.set_major_formatter('{x:1.0f}$')
        ax.legend(loc="best")
        return fig

    @staticmethod
    def draw_prediction_full(train_target, train_prediction, test_target, test_prediction, split_date='2022-02-22'):
        fig, ax = plt.subplots(1, 1)
        ax.plot(pd.concat([train_target, test_target]), label="Target",
                color=Visualizer.TARGET_COLOR)
        ax.plot(pd.concat([train_prediction, test_prediction]), label="Prediction",
                color=Visualizer.PREDICTION_COLOR)
        ax.tick_params(axis='x', labelrotation=90)
        ax.set(xlabel='Date', ylabel='Price in USD$')
        ax.yaxis.set_major_formatter('{x:1.0f}$')
        ax.legend(loc="best")
        ax.axvline(pd.to_datetime(split_date), color='r',
                   linestyle='--', label='Specific Date')
        return fig

    @staticmethod
    def get_missing_columns(data: pd.DataFrame) -> dict:
        return {col: [data[col].isnull().sum(),
                      f'% {np.round(np.mean(data[col].isnull()*100), 3)}'
                      ] for col in data.columns if data[col].isnull().any()}

    @staticmethod
    def draw_missing_data(data: pd.DataFrame):
        fig, ax = plt.subplots(1, 1, figsize=(18,8))
        sns.heatmap(data.isnull(), ax=ax,
                    cmap=sns.color_palette(['#34495E', 'tab:blue']))
        return fig
