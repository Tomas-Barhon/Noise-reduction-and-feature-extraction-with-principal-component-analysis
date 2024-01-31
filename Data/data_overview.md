# Data overview (legend)

## Studied cryptocurrencies
* Bitcoin 
* Ethereum
* Litecoin

## Types of variables
* macroeconomical
* technical (MA, returns, etc...)
* search realted
* speculative (fear index?)
* fundamental (hashrate, active adresses, etc...)

## Possible sources?
* Yahoo finance (prices,indexes)
* coinmetrics (crypto funadamental)
* Google trends
* Wiki searches

## Yahoo finance
### Dependent variables (crypto prices)
* "BTC-USD" - bitcoin price
* "ETH-USD" - ethereum price
* "LTC-USD" - Litecoin price

### Technical analysis variables (volatilities, moving averages, returns ...)

### Market state indexes
* "^DJI" - Dow Jones industrial average 
* "^GSPC" - S&P 500
* "GC=F" - Gold futures
### Market sentiment about volatility
* "^VIX" - CBOE Volatility Index 

### ETFs and Indexes orinted towards technological companies
**INDEXES**
* "^IXIC" - NASDAQ composite

**ETFS**
* "SMH" - VanEck Semiconductor ETF
* "VGT" - Vanguard Information Technology Index Fund
* "XSD" - SPDR S&P Semiconductor ETF
* "IYW" - iShares U.S. Technology ETF
* "FTEC" - Fidelity MSCI Information Technology Index ETF
* "IGV" - iShares Expanded Tech-Software Sector ETF
* "QQQ" - Invesco QQQ Trust


## Macroeconomical data

## Google trends
* Bitcoin
* Ethereum
* Litecoin
* Overall crypto search, to be included in all datasets

* Note: I had to remove terms BTC/LTC/ETH-USD as there is not enough data and these searches are not recognized by google trends
* TBD - measure of market sentiment (good vs bad types of search) (maybe just VIX + moving averages)
## Wiki searches individual
* page views - Bitcoin 
* page views - Ethereum
* page views - Litecoin
## Crypto fundamental

### **BTC** (30)
* BTC / Addresses, active, count
* BTC / NVT, adjusted, 90d MA
* BTC / NVT, adjusted, free float,  90d MA
* BTC / NVT, adjusted
* BTC / NVT, adjusted, free float
* BTC / Flow, in, to exchanges, USD
* BTC / Flow, out, from exchanges, USD
* BTC / Fees, transaction, mean, USD
* BTC / Fees, transaction, median, USD
* BTC / Fees, total, USD
* BTC / Miner revenue, USD
* BTC / Capitalization, market, free float, USD
* BTC / Capitalization, realized, USD
* BTC / Capitalization, market, current supply, USD
* BTC / Capitalization, market, estimated supply, USD
* BTC / Volatility, daily returns, 30d
* BTC / Volatility, daily returns, 180d
* BTC / Difficulty, last
* BTC / Difficulty, mean
* BTC / Hash rate, mean
* BTC / Hash rate, mean, 30d
* BTC / Revenue, per hash unit, USD
* BTC / Supply, Miner, held by all mining entities, USD
* BTC / Block, size, mean, bytes
* BTC / Block, weight, mean
* BTC / Issuance, continuous, percent, daily
* BTC / Network distribution factor
* BTC / Transactions, count
* BTC / Transactions, transfers, count
* BTC / Transactions, transfers, value, mean, USD

### **ETH** (27) - missing hashrate mean 30d; Supply, Miner, held by all mining entities, USD; Block, weight, mean
* ETH / Addresses, active, count
* ETH / NVT, adjusted, 90d MA
* ETH / NVT, adjusted, free float,  90d MA
* ETH / NVT, adjusted
* ETH / NVT, adjusted, free float
* ETH / Flow, in, to exchanges, USD
* ETH / Flow, out, from exchanges, USD
* ETH / Fees, transaction, mean, USD
* ETH / Fees, transaction, median, USD
* ETH / Fees, total, USD
* ETH / Miner revenue, USD
* ETH / Capitalization, market, free float, USD
* ETH / Capitalization, realized, USD
* ETH / Capitalization, market, current supply, USD
* ETH / Capitalization, market, estimated supply, USD
* ETH / Volatility, daily returns, 30d
* ETH / Volatility, daily returns, 180d
* ETH / Difficulty, last
* ETH / Difficulty, mean
* ETH / Hash rate, mean
* ETH / Revenue, per hash unit, USD
* ETH / Block, size, mean, bytes
* ETH / Issuance, continuous, percent, daily
* ETH / Network distribution factor
* ETH / Transactions, count
* ETH / Transactions, transfers, count
* ETH / Transactions, transfers, value, mean, USD


### **LTC** (25) - missing Flow, in, to exchanges, USD; Flow, out, from exchanges, USD; Revenue, per hash unit, USD; hashrate mean 30d; Supply, Miner, held by all mining entities, USD
* LTC / Addresses, active, count
* LTC / NVT, adjusted, 90d MA
* LTC / NVT, adjusted, free float,  90d MA
* LTC / NVT, adjusted
* LTC / NVT, adjusted, free float
* LTC / Fees, transaction, mean, USD
* LTC / Fees, transaction, median, USD
* LTC / Fees, total, USD
* LTC / Miner revenue, USD
* LTC / Capitalization, market, free float, USD
* LTC / Capitalization, realized, USD
* LTC / Capitalization, market, current supply, USD
* LTC / Capitalization, market, estimated supply, USD
* LTC / Volatility, daily returns, 30d
* LTC / Volatility, daily returns, 180d
* LTC / Difficulty, last
* LTC / Difficulty, mean
* LTC / Hash rate, mean
* LTC / Block, size, mean, bytes
* LTC / Block, weight, mean
* LTC / Issuance, continuous, percent, daily
* LTC / Network distribution factor
* LTC / Transactions, count
* LTC / Transactions, transfers, count
* LTC / Transactions, transfers, value, mean, USD



# Notes & ideas:

## google/wikipedia trends 

* (hard to distinguish between positive and negative interest -> introducing dummy variable to account for that)

* Google-trends,not case senstitive (different variation of the word)
* Google-trends normalized not the actual number
* Mention that because we will sum all the search possibilites there is a high risk of overlooking something and we also increase the errors in the confidence intervals
* How to construct overall market sentiment from Google search 
* Fix for the relative values appearing in Google trends https://github.com/epfl-dlab/GoogleTrendsAnchorBank

* MAKE SURE TO SHIFT THE DATA BEFORE INTERPOLATION !!!
For instance having monthly cpi data we need to shift it by 1 and then interpolate to daily data so that we avoid looking into the future !!!

* Do not forget on bootstrap resampling for model comparison

* It is crucial to figure out correct methodology/framework how to compare the models as the models with higher dimensionality data need more capacity than
the ones with lower dimensionality (hard to conclude a ceteris paribus effect of the reduciton)
* Scalers have to part of the pipeline, or one has to ensure that the scalers use only the training data distribution (we cannot leak the distributions of dev, test even to scalers!!!)



