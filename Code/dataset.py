import time
from abc import abstractmethod
from datetime import datetime
from fredapi import Fred
import pandas as pd
import yfinance as yf
import gtab
class Dataset():
    """
    Parent class for other datasets responsible for getting data
    for variables common in all datasets.
    """
    #definning constants with Tick names
    #targets
    BTC = "BTC-USD"
    ETH = "ETH-USD"
    LTC = "LTC-USD"

    # Market state indicators
    # Dow Jones industrial average
    DJI = "^DJI"
    # S&P 500
    SP500 = "^GSPC"
    # Gold futures
    GOLD_FUTURES = "GC=F"

    # Market sentiment about volatility
    # CBOE Volatility Index
    CBOE_VOLATILITY = "^VIX"

    # ETFs and Indexes orinted towards technological companies
    # NASDAQ composite
    NASDAQ = "^IXIC"
    # VanEck Semiconductor ETF
    SMH = "SMH"
    # Vanguard Information Technology Index Fund
    VGT = "VGT"
    # SPDR S&P Semiconductor ETF
    SPDR_SEMICONDUCTOR = "XSD"
    # iShares U.S. Technology ETF
    ISHARES = "IYW"
    # Fidelity MSCI Information Technology Index ETF
    FIDELITYMSCI = "FTEC"
    # iShares Expanded Tech-Software Sector ETF
    ISHARES_EXPANDED = "IGV"
    # Invesco QQQ Trust
    INVESCO_QQQ = "QQQ"

    # Fred St. Louis data
    # Real Gross Domestic product in the US
    RGDP_US = "GDPC1"
    # Real gross domestic product per capita in the US
    RGDP_PC_US = "A939RX0Q048SBEA"
    # Consumer Price Index: All Items: Total for United States
    CPI_US = "USACPALTT01CTGYM"
    # M2 base US
    M2_US = "WM2NS"
    # U.S. Dollars to Euro Spot Exchange Rate
    USD_EUR_rate = "DEXUSEU"
    

    def __init__(self):
        self.data = pd.DataFrame(columns=["no_data"])
        self.date_min = datetime(2011,1,3)
        self.date_max = datetime(2022,12,30)
        #normalized google trends
        self.google_trends = gtab.GTAB()
        self.google_trends.set_options(pytrends_config={"timeframe": "2011-01-01 2023-01-01"})
        self.google_trends.set_active_gtab("google_anchorbank_geo=_timeframe=2011-01-01 2023-01-01.tsv")
        self.queries_crypto = [self.google_trends.new_query("Cryptocurrency"),
                                self.google_trends.new_query("cryptocurrency"),
                                self.google_trends.new_query("Cryptocurrencies"),
                                self.google_trends.new_query("cryptocurrencies"),
                                self.google_trends.new_query("crypto"),
                                self.google_trends.new_query("Crypto")]
        self.fred = Fred(api_key="ee915eacae9f30debeafbd04ea173709")

    def get_yf_variable_history(self,tick_name):
        """Retrieves data from Yahoo finance and returns pd.Dataframe.
        """
        ticker = yf.Ticker(tick_name)
        history = ticker.history(period="max")
        history.index = pd.to_datetime(pd.to_datetime(history.index).date)
        return history["Close"]

    def create_coinmetrics_datasets(self):
        """Method that takes coinmetrics dataset and creates new csv file for each cryptocurrency
        """
        #the seperator is tab and encoding UTF-16-LE just so I do not forget
        data = pd.read_csv("./../Data/coinmetrics_data_4_12_2023.csv",
                        encoding="utf-16-le", sep = "	")
        data["Time"] = pd.to_datetime(data["Time"])
        data.set_index("Time", inplace=True)
        data.index.name = None
        #spliting data into 3 seperate datasets
        self.BTC_dataframe, self.ETH_dataframe, self.LTC_dataframe = (data.iloc[:,:30],
                                            data.iloc[:,30:57], data.iloc[:,57:])
        return self

    def get_common_data(self):
        """
        returns dataframe with variables that are common for all cryptocurrencies
        """
        #Yahoo finance tickers
        merged_df = self.get_yf_variable_history(Dataset.DJI)
        common_variables = [Dataset.SP500,Dataset.GOLD_FUTURES,Dataset.CBOE_VOLATILITY,
                            Dataset.NASDAQ,Dataset.SMH,Dataset.VGT,Dataset.SPDR_SEMICONDUCTOR,
                            Dataset.ISHARES,Dataset.FIDELITYMSCI,Dataset.ISHARES_EXPANDED,
                            Dataset.INVESCO_QQQ]
        for tick in common_variables:
            second = tick
            merged_df = pd.merge(merged_df, self.get_yf_variable_history(tick),
                                left_index=True, right_index=True,
                        how="outer", suffixes = (None, "_" + second))
        merged_df.rename(columns={"Close": "Close_^DJI"}, inplace=True)
        #Google crypto
        google_crypto = self.combine_queries(self.queries_crypto)
        google_crypto.rename(columns={"max_ratio": "Google_crypto_search"}, inplace=True)
        google_crypto = google_crypto[["Google_crypto_search"]]
        merged_df = pd.merge(merged_df, google_crypto,
                                left_index=True, right_index=True,
                        how="outer", suffixes = (None, None))
        #Wiki crypto
        wiki_crypto = pd.read_csv("./../Data/pageviews-Cryptocurrency.csv")
        wiki_crypto["Date"] = pd.to_datetime(wiki_crypto["Date"])
        wiki_crypto.set_index("Date", inplace=True)
        wiki_crypto.index.name = None
        wiki_crypto.rename(columns={"Cryptocurrency": "Wiki_crypto_search"}, inplace=True)
        merged_df = pd.merge(merged_df, wiki_crypto,
                                left_index=True, right_index=True,
                        how="outer", suffixes = (None, None))
        
        #Fred macro data, need to be interpolated to daily
        #macro_data = fred.get_series("CPIAUCSL")
        fred_data = self.get_fred_data()
        for column in fred_data:
            merged_df = pd.merge(merged_df, column,
                                left_index=True, right_index=True,
                        how="outer", suffixes = (None, None))
        #assign common data to self.data
        self.data = merged_df
        return self
        
    def get_data(self):
        return self.data

    def combine_queries(self, queries):
        """
        Function that will sum the different query formulation and return the summed dataframe
        """
        result = queries[0]
        for query in queries[1:]:
            result["max_ratio"] = result["max_ratio"] + query["max_ratio"]
        result.index = pd.to_datetime(result.index)
        result = result.resample("D").ffill()
        return result
    
    def get_fred_data(self):
        data_buffer = []
        ticks = [self.RGDP_US, self.RGDP_PC_US, self.CPI_US, self.M2_US, self.USD_EUR_rate]
        column_names = ["RGDP_US", "RGDP_PC_US", "CPI_US", "M2_US", "USD_EUR_rate"]
        for tick,name in zip(ticks, column_names):
            fred_series = self.fred.get_series(tick)
            fred_series.name = name
            fred_series.index = pd.to_datetime(fred_series.index)
            fred_series = fred_series.resample("D").ffill()
            data_buffer.append(fred_series.to_frame())
        return data_buffer
    
    def execute_full_pipeline(self):
        self.create_coinmetrics_datasets()
        self.get_common_data()
        return self

    @staticmethod
    def get_btc_data():
        dataset = pd.read_csv("./../Data/btc.csv", index_col=0)
        dataset.index = pd.to_datetime(dataset.index)
        return dataset

    @staticmethod
    def get_eth_data():
        dataset = pd.read_csv("./../Data/eth.csv", index_col=0)
        dataset.index = pd.to_datetime(dataset.index)
        return dataset

    @staticmethod
    def get_ltc_data():
        dataset = pd.read_csv("./../Data/ltc.csv", index_col=0)
        dataset.index = pd.to_datetime(dataset.index)
        return dataset

    @abstractmethod
    def merge_all_data(self):
        pass

    @abstractmethod
    def save_data_to_csv(self):
        pass
class BitcoinDataset(Dataset):
    def __init__(self):
        super().__init__()
        time.sleep(20)
        self.queries_bitcoin = [self.google_trends.new_query("Bitcoin"),
                                self.google_trends.new_query("bitcoin"),
                                self.google_trends.new_query("BTC")]

    def merge_all_data(self):
        self.execute_full_pipeline()
        #Currency specific google searches
        google_btc = self.combine_queries(self.queries_bitcoin)
        google_btc.rename(columns={"max_ratio": "Google_btc_search"}, inplace=True)
        google_btc = google_btc[["Google_btc_search"]]
        #Wiki btc
        wiki_crypto = pd.read_csv("./../Data/pageviews-Bitcoin.csv")
        wiki_crypto["Date"] = pd.to_datetime(wiki_crypto["Date"])
        wiki_crypto.set_index("Date", inplace=True)
        wiki_crypto.index.name = None
        wiki_crypto.rename(columns={"Bitcoin": "Wiki_btc_search"}, inplace=True)
        google_wiki = pd.merge(google_btc, wiki_crypto,
                                left_index=True, right_index=True,
                        how="outer", suffixes = (None, None))
        merged_df = pd.merge(self.BTC_dataframe, google_wiki,
                                left_index=True, right_index=True,
                        how="outer", suffixes = (None, None))
        merged_df = pd.merge(merged_df, self.data,
                                left_index=True, right_index=True,
                        how="outer", suffixes = (None, None))
        merged_df = pd.merge(merged_df, self.get_yf_variable_history("BTC-USD"),
                                left_index=True, right_index=True,
                        how="outer", suffixes = (None, None))
        merged_df.rename(columns={"Close": "BTC-USD"}, inplace=True)
        return merged_df
    def save_data_to_csv(self):
        data = self.merge_all_data()
        data = data.loc[self.date_min:self.date_max]
        data.to_csv("./../Data/btc.csv")
        return self
class EthereumDataset(Dataset):
    def __init__(self):
        super().__init__()
        time.sleep(20)
        self.queries_ethereum = [self.google_trends.new_query("Ethereum"),
                                self.google_trends.new_query("ethereum"),
                                self.google_trends.new_query("ether"),
                                self.google_trends.new_query("ETH")]

    def merge_all_data(self):
        self.execute_full_pipeline()
        #Currency specific google searches
        google_eth = self.combine_queries(self.queries_ethereum)
        google_eth.rename(columns={"max_ratio": "Google_eth_search"}, inplace=True)
        google_eth = google_eth[["Google_eth_search"]]
        #Wiki btc
        wiki_crypto = pd.read_csv("./../Data/pageviews-Ethereum.csv")
        wiki_crypto["Date"] = pd.to_datetime(wiki_crypto["Date"])
        wiki_crypto.set_index("Date", inplace=True)
        wiki_crypto.index.name = None
        wiki_crypto.rename(columns={"Ethereum": "Wiki_eth_search"}, inplace=True)
        google_wiki = pd.merge(google_eth, wiki_crypto,
                                left_index=True, right_index=True,
                        how="outer", suffixes = (None, None))
        merged_df = pd.merge(self.ETH_dataframe, google_wiki,
                                left_index=True, right_index=True,
                        how="outer", suffixes = (None, None))
        merged_df = pd.merge(merged_df, self.data,
                                left_index=True, right_index=True,
                        how="outer", suffixes = (None, None))
        merged_df = pd.merge(merged_df, self.get_yf_variable_history("ETH-USD"),
                                left_index=True, right_index=True,
                        how="outer", suffixes = (None, None))
        merged_df.rename(columns={"Close": "ETH-USD"}, inplace=True)
        return merged_df
    def save_data_to_csv(self):
        data = self.merge_all_data()
        data = data.loc[self.date_min:self.date_max]
        data.to_csv("./../Data/eth.csv")
        return self
class LitecoinDataset(Dataset):
    def __init__(self):
        super().__init__()
        time.sleep(20)
        self.queries_litecoin = [self.google_trends.new_query("Litecoin"),
                        self.google_trends.new_query("litecoin"),
                        self.google_trends.new_query("LTC")]

    def merge_all_data(self):
        self.execute_full_pipeline()
        #Currency specific google searches
        google_ltc = self.combine_queries(self.queries_litecoin)
        google_ltc.rename(columns={"max_ratio": "Google_ltc_search"}, inplace=True)
        google_ltc = google_ltc[["Google_ltc_search"]]
        #Wiki btc
        wiki_crypto = pd.read_csv("./../Data/pageviews-Litecoin.csv")
        wiki_crypto["Date"] = pd.to_datetime(wiki_crypto["Date"])
        wiki_crypto.set_index("Date", inplace=True)
        wiki_crypto.index.name = None
        wiki_crypto.rename(columns={"Litecoin": "Wiki_ltc_search"}, inplace=True)
        google_wiki = pd.merge(google_ltc, wiki_crypto,
                                left_index=True, right_index=True,
                        how="outer", suffixes = (None, None))
        merged_df = pd.merge(self.LTC_dataframe, google_wiki,
                                left_index=True, right_index=True,
                        how="outer", suffixes = (None, None))
        merged_df = pd.merge(merged_df, self.data,
                                left_index=True, right_index=True,
                        how="outer", suffixes = (None, None))
        merged_df = pd.merge(merged_df, self.get_yf_variable_history("LTC-USD"),
                                left_index=True, right_index=True,
                        how="outer", suffixes = (None, None))
        merged_df.rename(columns={"Close": "LTC-USD"}, inplace=True)
        return merged_df
    def save_data_to_csv(self):
        data = self.merge_all_data()
        data = data.loc[self.date_min:self.date_max]
        data.to_csv("./../Data/ltc.csv")
        return self