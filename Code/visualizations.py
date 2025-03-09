import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import sklearn.model_selection


class Visualizer():
    """
    Class encapsulating all the visualizations as static methods.
    """
    
    TARGET_COLOR = "black"
    PREDICTION_COLOR = "tab:blue"
    STYLE = "seaborn-v0_8-notebook"

    def __init__(self) -> None:
        plt.style.use(self.STYLE)

    @staticmethod
    def draw_prediction_test(test_target, test_prediction, shift):
        """
        Draws prediction on the test dataset against the target variables."""
        fig, ax = plt.subplots(1, 1)
        ax.plot(test_target.shift(shift), label="Target",
                color=Visualizer.TARGET_COLOR, linewidth=0.5)
        ax.plot(test_prediction.shift(shift), label="Prediction",
                color=Visualizer.PREDICTION_COLOR, linewidth=0.5)
        ax.tick_params(axis='x', labelrotation=90)
        ax.set_title(f"Price prediction {shift} days in advance", fontsize=14)
        ax.set(xlabel='Date', ylabel='Price in USD$')
        ax.yaxis.set_major_formatter('{x:1.0f}$')
        ax.legend(loc="best")
        return fig

    @staticmethod
    def draw_prediction_full(train_target, train_prediction, test_target,
                             test_prediction, shift, split_date='2022-02-22'):
        """
        Draws prediction for the whole dataset and indicates the split between training and testing data."""
        fig, ax = plt.subplots(1, 1)
        ax.plot(pd.concat([train_target, test_target]).shift(shift), label="Target",
                color=Visualizer.TARGET_COLOR, linewidth=0.5)
        ax.plot(pd.concat([train_prediction, test_prediction]).shift(shift), label="Prediction",
                color=Visualizer.PREDICTION_COLOR, linewidth=0.5)
        ax.tick_params(axis='x', labelrotation=90)
        ax.set(xlabel='Date', ylabel='Price in USD$')
        ax.set_title(f"Price prediction {shift} days in advance", fontsize=14)
        ax.yaxis.set_major_formatter('{x:1.0f}$')
        ax.axvline(pd.to_datetime(split_date), color='r',
                   linestyle='--', label="Train-test split")
        ax.legend(loc="best")
        return fig

    @staticmethod
    def get_missing_columns(data: pd.DataFrame) -> dict:
        """
        Returns a dictionary indicating percentage of missing values in each column
        of pd.DataFrame.
        """
        return {col: [data[col].isnull().sum(),
                      f'% {np.round(np.mean(data[col].isnull()*100), 3)}'
                      ] for col in data.columns if data[col].isnull().any()}

    @staticmethod
    def draw_missing_data(data: pd.DataFrame):
        """
        Creates a heatmap of missing values in the whole dataframe."""
        fig, ax = plt.subplots(1, 1, figsize=(18, 11))
        sns.heatmap(data.isnull(), ax=ax,
                    cmap=sns.color_palette(['#34495E', 'tab:blue']))
        ax.set_yticklabels([t.get_text().split("T")[0]
                           for t in ax.get_yticklabels()])
        plt.xticks(rotation=45, ha="right")
        return fig

    @staticmethod
    def draw_corr_cov_heatmap(data: pd.DataFrame, correlation=True):
        """
        Creates correlation or covariance matrix of data.
        """
        if correlation is True:
            corr = data.corr()
        else:
            corr = data.cov()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        fig, ax = plt.subplots(figsize=(11, 9))
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        sns.heatmap(corr, mask=mask, cmap=cmap, center=0,
                    square=True, linewidths=.5, cbar_kws={"shrink": .5})
        return fig

    @staticmethod
    def show_ts_split_shapes(data):
        """
        Returns a list of shapes that shows how does TimeSeriesSplit work
        """
        split = sklearn.model_selection.TimeSeriesSplit(n_splits=3)
        return [(el[0].shape, el[1].shape) for el in split.split(data)]

    @staticmethod
    def draw_cumulative_varience_ratios(data):
        """
        Draws cumulative variance per principal component.
        """
        fig, ax = plt.subplots(1, 1)
        ax.plot(np.arange(1, len(data) + 1).astype(np.int16), data*100,
                color=Visualizer.PREDICTION_COLOR, linewidth=1)
        ax.set_title(
            "Cumulative retained variance by principal components in %", fontsize=14)
        ax.set(xlabel="Number of principal components",
               ylabel="Retained variance ratio %")
        print(np.min(data))
        ax.set_ylim(50, 100)
        return fig
