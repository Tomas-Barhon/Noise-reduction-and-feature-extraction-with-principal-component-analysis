import matplotlib.pyplot as plt
import scienceplots
import pandas as pd


class Visualizer():
    def __init__(self) -> None:
        ...

    def draw_prediction_test(self):
        ...

    def draw_prediction_full(self, train_target, train_prediction, test_target, test_prediction, split_date = '2022-02-22'):
        fig, ax = plt.subplots(1, 1)
        ax.plot(pd.concat([train_target, test_target]), label="Target")
        ax.plot(pd.concat([train_prediction, test_prediction]), label="Prediction")
        ax.tick_params(axis='x', labelrotation = 90)
        ax.set(xlabel='Date', ylabel='Price in USD$')
        ax.yaxis.set_major_formatter('{x:1.0f}$')
        ax.legend(loc="best")
        ax.axvline(pd.to_datetime(split_date), color='r',
                   linestyle='--', label='Specific Date')
        return fig
