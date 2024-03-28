import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ta
import streamlit as st

# Function to fetch historical stock price data using Yahoo Finance API
def fetch_stock_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data


def calculate_technical_indicators(data, short_term_ema_days=100, long_term_ema_days=200):
    data['Short_EMA'] = data['Close'].ewm(span=short_term_ema_days, adjust=False).mean()
    data['Long_EMA'] = data['Close'].ewm(span=long_term_ema_days, adjust=False).mean()

    # Calculate RSI
    data['RSI'] = ta.momentum.RSIIndicator(data['Close'], window=14).rsi()

    return data


# Entry conditions based on EMAs and RSI
def entry_condition(data, current_index):
    if current_index < 1:
        return False
    return (data['Short_EMA'].iloc[current_index - 1] < data['Long_EMA'].iloc[current_index - 1]) and \
        (data['Short_EMA'].iloc[current_index] > data['Long_EMA'].iloc[current_index])
            #and \
            #(data['RSI'].iloc[current_index] >= 50) and (data['RSI'].iloc[current_index] <= 70))


# Exit conditions based on RSI and profit
def exit_condition(data, current_index):
    if current_index < 1:
        return False
    return (data['Short_EMA'].iloc[current_index - 1] > data['Long_EMA'].iloc[current_index - 1]) and \
        (data['Short_EMA'].iloc[current_index] < data['Long_EMA'].iloc[current_index])
            #and \
            #(data['RSI'].iloc[current_index] <= 50) and (data['RSI'].iloc[current_index] >= 30)

# Backtest strategy
def backtest_strategy(data, entry_condition, exit_condition, commission, initial_investment=400):
    position = None
    buy_price = None
    accumulated_profit = 0
    buy_markers = []
    sell_markers = []
    cumulative_profit = []

    shares_bought = 0  # Initialize shares_bought outside the loop

    for index, row in data.iterrows():
        close_price = row['Close']
        current_index = data.index.get_loc(index)  # Get the current index position

        # Check entry condition
        if entry_condition(data, current_index) and position is None:
            position = index
            buy_price = close_price
            buy_markers.append(index)
            shares_bought = initial_investment / buy_price

        # Check exit condition
        if exit_condition(data, current_index) and position is not None:
            sell_price = close_price
            profit_loss = (sell_price - buy_price) * shares_bought - commission  # Account for commission
            #if profit_loss > 0:  # Sell only if profit is positive
            accumulated_profit += profit_loss
            position = None
            sell_markers.append(index)

        cumulative_profit.append(accumulated_profit)

    return buy_markers, sell_markers, cumulative_profit




def main():
    st.set_page_config(layout="wide")

    st.title('Stock Analysis')

    tickers = st.sidebar.text_input('Enter Tickers (comma-separated)',
                                    'SE,NU,ANNX,HEIA.AS,ADBE,FSLR,AMAT,ADSK,GOOG,PYPL,MSFT,AMD,NVDA')
    tickers = [ticker.strip() for ticker in tickers.split(',')]

    start_date = pd.Timestamp('2014-01-01')
    end_date = pd.Timestamp('2024-01-01')

    st.sidebar.subheader('Select Parameters')
    short_term_ema_days = st.sidebar.number_input('Short Term EMA (days)', value=100, step=1)
    long_term_ema_days = st.sidebar.number_input('Long Term EMA (days)', value=200, step=1)
    commission = st.sidebar.number_input('Commission', value=10.0, step=1.0)
    initial_investment = st.sidebar.number_input('Initial Investment', value=400, step=1)

    dataframes = []
    buy_markers_list = []
    sell_markers_list = []
    cumulative_profit_list = []

    new_signals = {}

    for ticker in tickers:
        data = fetch_stock_data(ticker, start_date, end_date)
        data = calculate_technical_indicators(data, short_term_ema_days, long_term_ema_days)

        # Backtest strategy based on EMAs and RSI
        buy_markers, sell_markers, cumulative_profit = backtest_strategy(data, entry_condition, exit_condition,
                                                                         commission, initial_investment)
        dataframes.append(data)
        buy_markers_list.append(buy_markers)
        sell_markers_list.append(sell_markers)
        cumulative_profit_list.append(cumulative_profit)

        last_date = data.index[-1]
        new_buy_signal = len(buy_markers) > 0 and buy_markers[-1] == last_date
        new_sell_signal = len(sell_markers) > 0 and sell_markers[-1] == last_date

        new_signals[ticker] = {
            'Buy': new_buy_signal,
            'Sell': new_sell_signal
        }

    # Plot results for each ticker
    fig, ax1 = plt.subplots(figsize=(10, 6))

    for i, (dataframe, ticker) in enumerate(zip(dataframes, tickers)):
        color = plt.cm.tab10(i)  # Choose a color from the tab10 colormap
        linestyle = '-'  # Use a solid line style
        linewidth = 0.5  # Set line width to be thinner
        ax1.plot(dataframe.index, dataframe['Close'], label=f'{ticker} - Close Price', color=color, linestyle=linestyle,
                 linewidth=linewidth)
        ax1.plot(dataframe.index, dataframe['Short_EMA'], label=f'{ticker} - Short EMA', color=color, linestyle='--',
                 linewidth=linewidth)
        ax1.plot(dataframe.index, dataframe['Long_EMA'], label=f'{ticker} - Long EMA', color=color, linestyle=':',
                 linewidth=linewidth)

    ax1.set_ylabel('Price', fontsize=14)
    ax2 = ax1.twinx()

    for i, cumulative_profit in enumerate(cumulative_profit_list):
        color = plt.cm.tab10(i)  # Choose a color from the tab10 colormap
        linestyle = '-'  # Use a solid line style
        linewidth = 0.5  # Set line width to be thinner
        ax2.step(dataframes[i].index, cumulative_profit, label=f'{tickers[i]} - Cumulated Profit', color=color,
                 linestyle=linestyle, linewidth=linewidth)

    ax2.set_ylabel('Cumulated Profit', fontsize=14)
    ax1.set_title('Stock Prices and Cumulated Profits', fontsize=16)
    # ax1.legend(loc='upper left', fontsize=10)
    # ax2.legend(loc='upper right', fontsize=10)
    ax1.grid(True)
    ax2.grid(False)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()

    st.pyplot(fig)

    # Generate summary table
    table_data = []
    total_trades = 0
    total_avg_trades_per_week = 0
    total_num_wins = 0
    total_total_profit = 0
    total_avg_profit_per_month = 0
    total_avg_profit_per_trade = 0

    for ticker, cumulative_profit, sell_markers, buy_markers in zip(tickers, cumulative_profit_list, sell_markers_list, buy_markers_list):
        trades = len(sell_markers)
        avg_trades_per_week = trades / ((end_date - start_date).days / 7)
        num_wins = sum(1 for i in range(1, len(cumulative_profit)) if cumulative_profit[i] > cumulative_profit[i - 1])
        if cumulative_profit:
            total_profit = cumulative_profit[-1]
        else:
            total_profit = 0

        avg_profit_per_month = total_profit / (
                    (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month))
        avg_profit_per_trade = total_profit / trades if trades > 0 else 0  # Avoid division by zero
        total_trades += trades
        total_avg_trades_per_week += avg_trades_per_week
        total_num_wins += num_wins
        total_total_profit += total_profit
        total_avg_profit_per_month += avg_profit_per_month
        total_avg_profit_per_trade += avg_profit_per_trade

        table_data.append((ticker, trades, avg_trades_per_week, num_wins, total_profit, avg_profit_per_month,
                           round(avg_profit_per_trade, 2)))

    # Add row for total
    total_row = ('Total', total_trades, total_avg_trades_per_week, total_num_wins, total_total_profit,
                 total_avg_profit_per_month, round(total_avg_profit_per_trade, 2))
    table_data.append(total_row)

    # Display summary table
    st.subheader('Summary Table')
    df = pd.DataFrame(table_data,
                      columns=['Ticker', 'Number of Trades', 'Avg Trades/Week', 'Number of Wins', 'Total Profit',
                               'Avg Profit/Month', 'Avg Profit/Trade'])
    st.table(df)

    # Display new signals and open positions
    st.subheader('New Signals and Open Positions')

    # Create a list to store the data
    table_data = []

    for i, (ticker, signals) in enumerate(new_signals.items()):
        signal_status = ''
        if signals['Buy'] and signals['Sell']:
            signal_status = 'Both buy and sell signals'
        elif signals['Buy']:
            signal_status = 'Buy signal'
        elif signals['Sell']:
            signal_status = 'Sell signal'
        else:
            signal_status = 'No new buy or sell signals'

        # Check if there is an open position
        buy_markers = buy_markers_list[i]
        sell_markers = sell_markers_list[i]
        if len(buy_markers) > len(sell_markers):
            last_buy_date = buy_markers[-1]
            last_buy_price = dataframes[i].loc[last_buy_date, 'Close']
            current_close_price = dataframes[i].iloc[-1]['Close']
            shares_bought = initial_investment // last_buy_price  # Using integer division to get whole number of shares
            current_potential_profit = (current_close_price - last_buy_price) * shares_bought

            # Add data to the table
            table_data.append({
                'Ticker': ticker,
                'Signal': signal_status,
                'Open Position': True,
                'Buy Price': round(last_buy_price),
                'Last Close Price': round(current_close_price),
                'Current Potential Profit if Sold Today': round(current_potential_profit)
            })
        else:
            # Add data to the table
            table_data.append({
                'Ticker': ticker,
                'Signal': signal_status,
                'Open Position': False,
                'Buy Price': '',
                'Last Close Price': '',
                'Current Potential Profit if Sold Today': ''
            })

    # Convert the list of dictionaries to a DataFrame
    table_df = pd.DataFrame(table_data)

    # Display the table
    st.table(table_df)


if __name__ == "__main__":
    main()
