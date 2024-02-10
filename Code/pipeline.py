from typing import Literal
from dataset import Dataset


class Pipeline:
    def __init__(self, crypto_tick: Literal["btc", "ltc", "eth"]) -> None:
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

    def set_beginning(self, start_date: str):
        self.data = self.data.loc[start_date:]
        return self

    def preprocess_dataset(self):
        match self.tick:
            case "btc":
                #Filling crypto wiki searches with zeros as the data started to be collected later
                self.data['Wiki_crypto_search'] = self.data['Wiki_crypto_search'].fillna(0)
                self.data['Wiki_btc_search'] = self.data['Wiki_btc_search'].fillna(0)
                #also filling with zeros as the metric was not present at that time
                self.data["BTC / Capitalization, market, estimated supply, USD"] = self.data["BTC / Capitalization, market, estimated supply, USD"].fillna(0)
                #Forward filling all the indexes into weekends and other day when the maret was closed
                self.data[['Close_^DJI','Close_^GSPC','Close_GC=F','Close_^VIX','Close_^IXIC',
                    'Close_SMH','Close_VGT','Close_XSD','Close_IYW','Close_FTEC','Close_IGV',
                    'Close_QQQ','USD_EUR_rate']] = self.data[['Close_^DJI','Close_^GSPC','Close_GC=F','Close_^VIX','Close_^IXIC',
                    'Close_SMH','Close_VGT','Close_XSD','Close_IYW','Close_FTEC','Close_IGV',
                    'Close_QQQ','USD_EUR_rate']].ffill()
            case "ltc":
                ...
            case "eth":
                ...
        return self

    def shift_target(self):
        #1 day forecasting
        self.data_1d_shift = self.data.copy()
        self.data_1d_shift.iloc[:, -1] = self.data_1d_shift.iloc[:, -1].shift(-1)
        self.data_1d_shift = self.data_1d_shift.dropna()
        #5 day forecasting
        self.data_5d_shift = self.data.copy()
        self.data_5d_shift.iloc[:, -1] = self.data_5d_shift.iloc[:, -1].shift(-5)
        self.data_5d_shift = self.data_5d_shift.dropna()
        #10 day forecasting
        self.data_10d_shift = self.data.copy()
        self.data_10d_shift.iloc[:, -1] = self.data_10d_shift.iloc[:, -1].shift(-10)
        self.data_10d_shift = self.data_10d_shift.dropna()
        return self