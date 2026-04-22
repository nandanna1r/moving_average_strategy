import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# basic data acquuisition from Yahoo Finance
ticker = "SPY"
hist_period = "5y"

tick_obj = yf.Ticker(ticker)


price_data = tick_obj.history(period=hist_period)
print(f"Gathering data for {ticker} from Yahoo Finance....")

close_prices = price_data[['Close']].copy()

print("Data loaded. Here is a sample view:")
print(close_prices.head())


# calculating moving averages
short_window = 50
long_window = 200

close_prices['short_ma'] = close_prices['Close'].rolling(window=short_window).mean()
close_prices['long_ma'] = close_prices['Close'].rolling(window=long_window).mean()

close_prices['signal'] = np.where(close_prices['short_ma'] > close_prices['long_ma'], 1, 0)
close_prices['position_change'] = close_prices['signal'].diff()

close_prices.dropna(inplace=True)

print("Moving averages calculated. Here is a sample view:")
print(close_prices.tail())

# backtesting and eval
close_prices['market_returns'] = close_prices['Close'].pct_change()
close_prices['strat_returns'] = close_prices['market_returns'] * close_prices['position_change'].shift(1)

close_prices['cum_market_returns'] = (1 + close_prices['market_returns']).cumprod()
close_prices['cum_strat_returns'] = (1 + close_prices['strat_returns']).cumprod()

close_prices.dropna(inplace=True) 

close_prices['cum_market_returns'] = (1 + close_prices['market_returns']).cumprod()
close_prices['cum_strat_returns'] = (1 + close_prices['strat_returns']).cumprod()

rf_rate = 0.0 
daily_rf_rate = rf_rate / 252
excess_ret = close_prices['strat_returns'] - daily_rf_rate

sharpe_ratio = (excess_ret.mean() / excess_ret.std()) * np.sqrt(252)

print("\nBacktest Results:")
print(f"Total Market Return: {(close_prices['cum_market_returns'].iloc[-1] - 1) * 100:.2f}%")
print(f"Total Strategy Return: {(close_prices['cum_strat_returns'].iloc[-1] - 1) * 100:.2f}%")
print(f"Sharpe Ratio of Strategy: {sharpe_ratio:.2f}")

# visualization
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

# baseline lines
ax1.plot(close_prices.index, close_prices['Close'], label='SPY Price', color='black', alpha=0.4)
ax1.plot(close_prices.index, close_prices['short_ma'], label='50-Day SMA', color='blue')
ax1.plot(close_prices.index, close_prices['long_ma'], label='200-Day SMA', color='red')

# 3. Isolate the exact days we bought and sold
buys = close_prices[close_prices['position_change'] == 1.0]
sells = close_prices[close_prices['position_change'] == -1.0]

# buy and sell markers
ax1.scatter(buys.index, buys['Close'], marker='^', color='green', s=150, label='Buy Signal', zorder=5)
ax1.scatter(sells.index, sells['Close'], marker='v', color='red', s=150, label='Sell Signal', zorder=5)

ax1.set_title('SPY - 50/200 Day Moving Average Crossover')
ax1.set_ylabel('Price ($)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# money made vs the market
ax2.plot(close_prices.index, close_prices['cum_market_returns'], label='Market Return (Buy & Hold)', color='gray')
ax2.plot(close_prices.index, close_prices['cum_strat_returns'], label='Strategy Return', color='purple')

ax2.set_title('Cumulative Performance')
ax2.set_ylabel('Cumulative Return')
ax2.set_xlabel('Date')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()