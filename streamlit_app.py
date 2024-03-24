import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ta  # Import the ta module for calculating technical indicators
import streamlit as st

profit_losses = None  # Initialize profit_losses as a global variable


def fetch_stock_data(ticker, start_date, end_date):
    """Fetch historical stock price data using Yahoo Finance API."""
    data = yf.download(ticker, start=start_date, end=end_date)
    return data


def calculate_technical_indicators(data):
    """Calculate technical indicators including RSI, Stochastic Oscillator, Williams %R, 20-day MA, 50-day MA, and 200-day MA."""
    # Calculate RSI
    data['RSI'] = ta.momentum.RSIIndicator(data['Close'], window=14).rsi()

    # Calculate Stochastic Oscillator
    data['%K'] = ta.momentum.StochasticOscillator(data['High'], data['Low'], data['Close'], window=14, smooth_window=3).stoch()

    # Create WilliamsRIndicator object
    williams_r = ta.momentum.WilliamsRIndicator(data['High'], data['Low'], data['Close'], lbp=5)

    # Access Williams %R values directly from the object
    data['Williams_R'] = williams_r.williams_r()

    # Calculate 20-day moving average
    data['20_SMA'] = data['Close'].rolling(window=20).mean()

    # Calculate 50-day moving average
    data['50_SMA'] = data['Close'].rolling(window=50).mean()

    # Calculate 200-day moving average
    data['200_SMA'] = data['Close'].rolling(window=200).mean()

    return data.ffill()  # Forward fill NaN values




def backtest_strategy(data):
    """Backtest the trading strategy with modified entry and exit rules."""
    global profit_losses
    profit_losses = np.full(len(data), np.nan)  # Initialize with NaN values

    position = None
    buy_price = None
    accumulated_profit = 0
    buy_markers = []
    sell_markers = []

    for index, row in data.iterrows():
        close_price = row['Close']
        williams_r = row['Williams_R']

        # Get previous row's Williams %R value
        prev_williams_r = data.loc[index - pd.Timedelta(days=1), 'Williams_R'] if index - pd.Timedelta(
            days=1) in data.index else None

        # Buy Signal: Williams %R moves above -80
        if williams_r > -80 and prev_williams_r is not None and prev_williams_r <= -80:
            if position is None:
                position = index
                buy_price = close_price
                buy_markers.append(index)

        # Sell Signal: Williams %R moves below -20
        if williams_r < -20 and prev_williams_r is not None and prev_williams_r >= -20:
            if position is not None:
                profit_loss = close_price - buy_price
                accumulated_profit += profit_loss
                profit_losses[data.index.get_loc(index)] = accumulated_profit
                position = None
                sell_markers.append(index)

    # Fill NaN values with the last known profit value (to create step function)
    profit_losses = pd.Series(profit_losses).fillna(method='ffill').values

    # Pad zeros before the first sell signal
    first_sell_index = data.index.get_loc(sell_markers[0]) if sell_markers else len(data)
    profit_losses[:first_sell_index] = 0

    return buy_markers, sell_markers



def plot_results(data, buy_markers, sell_markers):
    """Plot stock price and cumulative profit/loss with buy/sell labels."""
    global profit_losses
    fig, ax1 = plt.subplots(figsize=(14, 7))

    # Plot stock price and moving averages
    ax1.plot(data.index, data['Close'], label='Close Price', color='blue')
    ax1.plot(data.index, data['20_SMA'], label='20-Day SMA', color='orange')
    ax1.plot(data.index, data['50_SMA'], label='50-Day SMA', color='green')

    # Plot the 200-day moving average
    ax1.plot(data.index, data['200_SMA'], label='200-Day SMA', color='red')

    # Fill the area between the 20-day and 200-day moving averages
    ax1.fill_between(data.index, data['20_SMA'], data['200_SMA'], where=data['20_SMA'] > data['200_SMA'], color='green', alpha=0.3)

    ax1.set_ylabel('Price')
    ax1.set_xlabel('Date')

    # Plot buy and sell markers
    ax1.scatter(buy_markers, data['Close'].loc[buy_markers], color='green', marker='^', label='Buy')
    ax1.scatter(sell_markers, data['Close'].loc[sell_markers], color='red', marker='v', label='Sell')

    # Plot profit/loss as step function at sell signals
    ax2 = ax1.twinx()
    ax2.step(data.index, profit_losses, where='post', label='Profit/Loss', color='orange')
    ax2.set_ylabel('Profit/Loss')

    fig.tight_layout()
    fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.9))

    plt.show()


if __name__ == "__main__":
    ticker = 'GOOG'  # Replace 'GOOG' with the ticker symbol of the stock you want to analyze
    start_date = '2016-01-01'  # Specify start date
    end_date = '2024-01-01'  # Specify end date

    data = fetch_stock_data(ticker, start_date, end_date)
    data = calculate_technical_indicators(data)  # Ensure this function is called first
    buy_markers, sell_markers = backtest_strategy(data)
    plot_results(data, buy_markers, sell_markers)

st.write("hello")