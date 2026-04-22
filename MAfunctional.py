import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def fetch_data(tick_sym: str, hist_period: str = "5y"):
    print(f"\nFetching {hist_period}s of data for {tick_sym}...")
    try:
        tick_obj = yf.Ticker(tick_sym)
        price_df = tick_obj.history(period=hist_period)
        
        if price_df.empty:
            raise ValueError(f"No data found for ticker '{tick_sym}'.")
            
        close_df = price_df[['Close']].copy()
        return close_df
    
    except Exception as e:
        print(f"Error fetching data: {e}")
        return pd.DataFrame()

def calc_signals(df: pd.DataFrame, short_win: int, long_win: int):
   
    df['short_ma'] = df['Close'].rolling(window=short_win).mean()
    df['long_ma'] = df['Close'].rolling(window=long_win).mean()

    df['signal'] = np.where(df['short_ma'] > df['long_ma'], 1.0, 0.0)
    
    df['pos_change'] = df['signal'].diff()
    
    df.dropna(inplace=True)
    return df

def calc_metrics(df: pd.DataFrame, rf_rate: float = 0.0):
  
    df['mkt_ret'] = df['Close'].pct_change()
    df['strat_ret'] = df['mkt_ret'] * df['signal'].shift(1)
    df.dropna(inplace=True) 

    df['cum_mkt_ret'] = (1 + df['mkt_ret']).cumprod()
    df['cum_strat_ret'] = (1 + df['strat_ret']).cumprod()

    daily_rf = rf_rate / 252
    excess_ret = df['strat_ret'] - daily_rf
    
    std_dev = excess_ret.std()
    if std_dev == 0:
        sharpe = 0.0
    else:
        sharpe = (excess_ret.mean() / std_dev) * np.sqrt(252)

    return {
        "mkt_total": (df['cum_mkt_ret'].iloc[-1] - 1) * 100,
        "strat_total": (df['cum_strat_ret'].iloc[-1] - 1) * 100,
        "sharpe_ratio": sharpe
    }

def plot_results(df: pd.DataFrame, tick_sym: str, short_win: int, long_win: int):
 
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    ax1.plot(df.index, df['Close'], label=f'{tick_sym} Price', color='black', alpha=0.4)
    ax1.plot(df.index, df['short_ma'], label=f'{short_win}-Day SMA', color='blue')
    ax1.plot(df.index, df['long_ma'], label=f'{long_win}-Day SMA', color='red')

    buys = df[df['pos_change'] == 1.0]
    sells = df[df['pos_change'] == -1.0]

    ax1.scatter(buys.index, buys['Close'], marker='^', color='green', s=150, label='Buy', zorder=5)
    ax1.scatter(sells.index, sells['Close'], marker='v', color='red', s=150, label='Sell', zorder=5)

    ax1.set_title(f'{tick_sym} - {short_win}/{long_win} Moving Average Crossover')
    ax1.set_ylabel('Price ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(df.index, df['cum_mkt_ret'], label='Market Return (Buy & Hold)', color='gray')
    ax2.plot(df.index, df['cum_strat_ret'], label='Strategy Return', color='purple')

    ax2.set_title('Cumulative Performance')
    ax2.set_ylabel('Cumulative Return')
    ax2.set_xlabel('Date')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def main():
   
    print("Simple Moving Average Strategy Backtester")
    
    while True:
        
        tick_sym = input("\nEnter Ticker Symbol (e.g., SPY, AAPL, BTC-USD) or 'q' to quit: ").upper()
        if tick_sym == 'Q':
            print("Exiting program. Goodbye!")
            break
            
        try:
            short_win = int(input("Enter Short MA Window (e.g., 50): "))
            long_win = int(input("Enter Long MA Window (e.g., 200): "))
            
            if short_win >= long_win:
                print("Error: Ensure Short MA < Long MA. Try again.")
                continue

        except ValueError:
            print("Error: Invalid ticker or input. Enter valid integers for windows. Try again.")
            continue

        price_df = fetch_data(tick_sym)
        if price_df.empty:
            continue
            
        price_df = calc_signals(price_df, short_win, long_win)
        
        if price_df.empty:
             print(f"Error: Not enough historical data to calculate the {long_win}-day moving average.")
             continue

        metrics = calc_metrics(price_df)

        print("\n--- Backtest Results ---")
        print(f"Market Return:   {metrics['mkt_total']:.2f}%")
        print(f"Strategy Return: {metrics['strat_total']:.2f}%")
        print(f"Sharpe Ratio:    {metrics['sharpe_ratio']:.2f}")

        plot_results(price_df, tick_sym, short_win, long_win)

if __name__ == "__main__":
    main()