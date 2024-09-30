"""
This module needs to be run the first time using the project if the datasets are missing
in order to download all the that from the sources and merging them into 
one dataframe per ticker.
"""

import time
from dataset import BitcoinDataset, EthereumDataset, LitecoinDataset



#Adding sleeps to surpass google trends request limit
btc_dataset = BitcoinDataset()
print("BTC dataset initialized, sleeping for 30s")
time.sleep(30)
eth_dataset = EthereumDataset()
print("eth dataset initialized, sleeping for 30s")
time.sleep(30)
ltc_dataset = LitecoinDataset()
print("eth dataset initialized")
print("Creating btc.csv")
btc_dataset.save_data_to_csv()
time.sleep(30)
print("Creating eth.csv")
eth_dataset.save_data_to_csv()
time.sleep(30)
print("Creating ltc.csv")
ltc_dataset.save_data_to_csv()
