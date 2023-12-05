"""
pylint
"""

from datetime import datetime
from fredapi import Fred
import pandas as pd
import yfinance as yf
import gtab

class Dataset():
    """_summary_
    Parent class for other datasets responsible for getting data
    for variables common in all datasets.
    Returns:
        _type_: _description_
    """
    #definning constants with Tick names
    BTC = "BTC-USD"
    ETH = "ETH-USD"
    BNB = "BNB-USD"

    DJI = "^DJI"
    SP500 = "^GSPC"
    CBOE_VOLATILITY = "^VIX"
    NASDAQ = "^IXIC"
    SMH = "SMH"
    VGT = "VGT"
    SPDR_SEMICONDUCTOR = "XSD"
    ISHARES = "IYW"
    FIDELITYMSCI = "FTEC"
    ISHARES_EXPANDED = "IGV"
    INVESCO_QQQ = "QQQ"
    GOLD_FUTURES = "GC=F"

    def __init__(self):
        self.data = pd.DataFrame(columns=["no_data"])
        self.date_min = datetime(2011,12,31)
        self.data_max = datetime(2022,5,6)
        self.google_trends = gtab.GTAB()
    def get_yf_variable_history(self,tick_name):
        """Retrieves data from Yahoo finance and returns pd.Dataframe.

        Args:
            tick_name (_type_): _description_

        Returns:
            _type_: _description_
        """
        ticker = yf.Ticker(tick_name)
        return ticker.history(period="max")
    def create_coinmetrics_datasets(self):
        """Method that takes coinmetrics dataset and creates new csv file for each cryptocurrency
        """
        #the seperator is tab and encoding UTF-16-LE just so I do not forget
        data = pd.read_csv("./../Data/coinmetrics_data_4_12_2023.csv", encoding="utf-16-le", sep = "	")
        data["Time"] = pd.to_datetime(data["Time"])
        data.set_index("Time", inplace=True)
        data.index.name = None
        #spliting data into 3 datasets
        BTC_dataframe, ETH_dataframe, LTC_dataframe = data.iloc[:,:30], data.iloc[:,30:57], data.iloc[:,57:]
        BTC_dataframe.to_csv("./../Data/BTC_dataframe.csv")
        ETH_dataframe.to_csv("./../Data/ETH_dataframe.csv")
        LTC_dataframe.to_csv("./../Data/LTC_dataframe.csv")
        
    def get_common_data(self):
        """
        returns dataframe with variables that are common for all cryptocurrencies
        """
        #Yahoo finance tickers
        merged_df = self.get_yf_variable_history(Dataset.DJI)
        common_variables = [Dataset.SP500,Dataset.CBOE_VOLATILITY,
                            Dataset.NASDAQ,Dataset.SMH,Dataset.VGT,Dataset.SPDR_SEMICONDUCTOR,
                            Dataset.ISHARES,Dataset.FIDELITYMSCI,Dataset.ISHARES_EXPANDED,
                            Dataset.INVESCO_QQQ,Dataset.GOLD_FUTURES]
        for tick in common_variables:
            merged_df = pd.merge(merged_df, self.get_yf_variable_history(tick),
                                left_index=True, right_index=True, how='outer')
        #Google market
        
        #Fred macro data, need to be interpolated to daily 
        fred = Fred(api_key="ee915eacae9f30debeafbd04ea173709")
        macro_data = fred.get_series("CPIAUCSL")
        return merged_df
    
    def data_to_csv(self, name):
        """Saves data into csv file.

        Args:
            name (_type_): _description_
        """
        self.data.to_csv(name)
        
    def get_data(self):
        return self.data

class BitcoinDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.queries_bitcoin = [self.google_trends.new_query("Bitcoin"),
                                self.google_trends.new_query("bitcoin"),
                                self.google_trends.new_query("BTC"),
                                self.google_trends.new_query("BTC-USD")]
    def get_google_trends(self):
        """
        Function that will sum the different query formulation and return the summed dataframe
        """
        ...
        
class EthereumDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.queries_ethereum = [self.google_trends.new_query("Ethereum"),
                                self.google_trends.new_query("ethereum"),
                                self.google_trends.new_query("ether"),
                                self.google_trends.new_query("ETH"),
                                self.google_trends.new_query("ETH-USD")]
        
class BNBDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.queries_litecoin = [self.google_trends.new_query("Liteccoin"),
                        self.google_trends.new_query("litecoin"),
                        self.google_trends.new_query("LTC"),
                        self.google_trends.new_query("LTC-USD")]