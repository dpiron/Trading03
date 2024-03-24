import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ta
import streamlit as st

profit_losses = None


def fetch_stock_data(ticker, start_date, end_date):
    """Fetch historical stock price data using Yahoo Finance API."""
    data = yf.download(ticker, start=start_date, end=end_date)
    return data


def calculate_technical_indicators(data):
    """Calculate technical indicators including RSI, Stochastic Oscillator, Williams %R, 50-day MA, and 200-day MA."""
    data['RSI'] = ta.momentum.RSIIndicator(data['Close'], window=14).rsi()
    data['%K'] = ta.momentum.StochasticOscillator(data['High'], data['Low'], data['Close'], window=14,
                                                  smooth_window=3).stoch()
    data['Williams_R'] = ta.momentum.WilliamsRIndicator(data['High'], data['Low'], data['Close'], lbp=5).williams_r()

    # Calculate 50-day and 200-day moving averages
    data['50_MA'] = data['Close'].rolling(window=50).mean()
    data['200_MA'] = data['Close'].rolling(window=200).mean()

    return data


def backtest_strategy(data):
    """Backtest the trading strategy with modified entry and exit rules."""
    global profit_losses
    profit_losses = np.full(len(data), np.nan)

    position = None
    buy_price = None
    accumulated_profit = 0
    buy_markers = []
    sell_markers = []

    for index, row in data.iterrows():
        close_price = row['Close']
        williams_r = row['Williams_R']
        prev_williams_r = data.loc[index - pd.Timedelta(days=1), 'Williams_R'] if index - pd.Timedelta(
            days=1) in data.index else None

        # Check for downtrend (50-day MA below 200-day MA)
        if row['50_MA'] < row['200_MA']:
            if position is not None:
                # Generate a sell signal if already in position
                profit_loss = close_price - buy_price
                accumulated_profit += profit_loss
                profit_losses[data.index.get_loc(index)] = accumulated_profit
                position = None
                sell_markers.append(index)
            continue  # Skip trading further until a buy signal

        if williams_r > -80 and prev_williams_r is not None and prev_williams_r <= -80:
            if position is None and index in data.index:
                position = index
                buy_price = close_price
                buy_markers.append(index)

        if williams_r < -20 and prev_williams_r is not None and prev_williams_r >= -20:
            if position is not None:
                profit_loss = close_price - buy_price
                accumulated_profit += profit_loss
                profit_losses[data.index.get_loc(index)] = accumulated_profit
                position = None
                sell_markers.append(index)

    profit_losses = pd.Series(profit_losses).fillna(method='ffill').values
    first_sell_index = data.index.get_loc(sell_markers[0]) if sell_markers else len(data)
    profit_losses[:first_sell_index] = 0

    return buy_markers, sell_markers



def plot_results(data, buy_markers, sell_markers, start_date, end_date):
    """Plot stock price and cumulative profit/loss with buy/sell labels."""
    global profit_losses
    fig, ax1 = plt.subplots(figsize=(20, 12))  # Increased figure size

    # Filter data within the specified date range
    filtered_data = data.loc[start_date:end_date]

    ax1.plot(filtered_data.index, filtered_data['Close'], label='Close Price', color='blue')
    ax1.plot(filtered_data.index, filtered_data['50_MA'], label='50-day MA', color='orange')
    ax1.plot(filtered_data.index, filtered_data['200_MA'], label='200-day MA', color='green')

    # Fill between 50-day and 200-day MAs
    ax1.fill_between(filtered_data.index, filtered_data['50_MA'], filtered_data['200_MA'],
                     where=filtered_data['50_MA'] < filtered_data['200_MA'], color='red', alpha=0.3)

    ax1.set_ylabel('Price', fontsize=16)  # Increased font size
    ax1.set_xlabel('Date', fontsize=16)  # Increased font size

    buy_markers_filtered = [marker for marker in buy_markers if marker in filtered_data.index]
    ax1.scatter(buy_markers_filtered, filtered_data['Close'].loc[buy_markers_filtered], color='green', marker='^',
                label='Buy', s=100)  # Increased marker size

    sell_markers_filtered = [marker for marker in sell_markers if marker in filtered_data.index]
    ax1.scatter(sell_markers_filtered, filtered_data['Close'].loc[sell_markers_filtered], color='red', marker='v',
                label='Sell', s=100)  # Increased marker size

    if profit_losses is not None and len(profit_losses) == len(data.index):
        ax2 = ax1.twinx()
        ax2.step(data.index, profit_losses, where='post', label='Profit/Loss',color='orange')
        ax2.set_ylabel('Profit/Loss', fontsize=16)  # Increased font size

    fig.tight_layout()
    fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.9), fontsize=14)  # Increased font size

    # Increase tick label sizes
    ax1.tick_params(axis='both', which='major', labelsize=14)
    if 'ax2' in locals():
        ax2.tick_params(axis='both', which='major', labelsize=14)

    return fig


if __name__ == "__main__":
    st.set_page_config(layout="wide", page_title='Stock Analysis', page_icon=":chart_with_upwards_trend:",
                       initial_sidebar_state="expanded")

    st.title('Stock Analysis')

    # Add input box for ticker symbol
    ticker = st.sidebar.text_input('Enter Ticker Symbol', 'GOOG')

    st.sidebar.title('Graph Settings')
    start_date_graph = st.sidebar.date_input('Start Date', pd.to_datetime('2014-01-01'),
                                             min_value=pd.to_datetime('2004-01-01'), max_value=pd.to_datetime('today'))
    end_date_graph = st.sidebar.date_input('End Date', pd.to_datetime('2024-01-01'))

    data = fetch_stock_data(ticker, start_date_graph, end_date_graph)
    data = calculate_technical_indicators(data)
    buy_markers, sell_markers = backtest_strategy(data)
    selected_data = data.loc[start_date_graph:end_date_graph]

    fig = plot_results(data, buy_markers, sell_markers, start_date_graph, end_date_graph)
    st.pyplot(fig)
