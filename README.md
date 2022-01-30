# Bot Genesis

## Steps of Code:
* Step 1:
  * Pick data from coinmarketcap.com (coins with market capitalization above 500 Mn):
    * Max data for total market capitalization
    * Daily market capitalization for 7 days
    * 7 day price data for these coins
    * Daily trade volume
    * Name of the coin
    * Rank of the coin
* Step 2:
  * Calculate 7 days avg percent change and absolute percent change for all the coins
  * Arrange them in ascending and descending orders
* Step 3:
  * Taking the average and absolute change in total market capitalization as the reference calculate the deviation with average and absolute change of that coin price.
  * filter the coins with deviation 10% or greater
  * filter the top 5 market cap coins
* Step 4:
  * Pont these coins based on their market cap, magnitude of deviation, trend of deviation and daily volume on avialable exchange (binance)

## Requirements:
1. ***python-coinmarketcap***  
Command : `pip install python-coinmarketcap`

## References:
1. Mark Down cheat-sheet : [Link 1](https://www.markdownguide.org/cheat-sheet/) , [Link 2](https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet) 
2. Coinmarketcap API: [Link 1](https://algotrading101.com/learn/coinmarketcap-api-guide/) , [Link 2](https://coinmarketcap.com/api/documentation/v1/#section/Introduction)
