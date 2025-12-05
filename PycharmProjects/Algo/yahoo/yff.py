import yfinance as yf
import pandas as pd
import numpy as np
import mplfinance as mpf
import vectorbt as vbt


def get_nq_1h():
    """Get 1-hour NQ futures data"""
    ticker = yf.Ticker("NQ=F")
    hourly = ticker.history(period="max", interval="1h")
    return hourly


def calculate_thresholds(df, vol_lookback=20, chart_res_minutes=60):
    # for 1-hour data
    # Trading hours per day: 6.5 hours (390 minutes)
    # Trading days per year: 252
    # Hours per year: 252 * 6.5 = 1638 hours
    periods_per_year = (252.0 * 6.5)
    # periods_per_year = (252.0 * 390.0) / chart_res_minutes

    df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))

    # Calculate annualized volatility
    # Standard deviation of log returns * sqrt(periods per year)
    df['volatility'] = df['log_return'].rolling(window=vol_lookback).std() * np.sqrt(periods_per_year)

    # Expected move for one bar - calculated based on current bar
    df['expected_move'] = df['Close'] * df['volatility'] * np.sqrt(1.0 / periods_per_year)

    # Define thresholds around current bar's close
    df['upper_threshold'] = df['Close'].shift(1) + df['expected_move'].shift(1)
    df['lower_threshold'] = df['Close'].shift(1) - df['expected_move'].shift(1)

    return df


def detect_crossover_signals(df):
    """
    Detect crossover signals with realistic execution:
    - Signal detected at bar close
    - Entry executed at NEXT bar's open
    """
    prev_close = df['Close'].shift(1)
    prev_upper = df['upper_threshold'].shift(1)
    prev_lower = df['lower_threshold'].shift(1)

    curr_close = df['Close']
    curr_upper = df['upper_threshold']
    curr_lower = df['lower_threshold']

    # Signal detected at current bar's close (as Series, not bool)
    df['long_signal'] = (prev_close <= prev_upper) & (curr_close > curr_upper)
    df['short_signal'] = (prev_close >= prev_lower) & (curr_close < curr_lower)

    # Shift signals forward by 1 bar = entry happens on next bar
    df['long_entry'] = df['long_signal'].shift(1, fill_value=False)
    df['short_entry'] = df['short_signal'].shift(1, fill_value=False)

    # Entry price is this bar's Open
    df['entry_price'] = np.nan
    df.loc[df['long_entry'], 'entry_price'] = df.loc[df['long_entry'], 'Open']
    df.loc[df['short_entry'], 'entry_price'] = df.loc[df['short_entry'], 'Open']

    # Position: 1 = Long, -1 = Short, 0 = No entry this bar
    df['position'] = 0
    df.loc[df['long_entry'], 'position'] = 1
    df.loc[df['short_entry'], 'position'] = -1

    return df

# Get and save 1-hour data
# data = get_nq_1h()
# data.to_csv("NQ_1h.csv")

df = pd.read_csv('NQ_1h.csv', index_col=0, parse_dates=True)
df['sma20'] = df['Close'].rolling(20).mean()
df = calculate_thresholds(df, 20)
df = detect_crossover_signals(df)


df.index = pd.to_datetime(df.index, utc=True).tz_convert(None)
df['long_marker'] = np.where(df['long_entry'], df['Low'] * 0.999, np.nan)
df['short_marker'] = np.where(df['short_entry'], df['High'] * 1.001, np.nan)
plot_df = df.tail(100).copy()

# Create additional plots (overlays)
add_plots = [
    mpf.make_addplot(plot_df['sma20'], color='blue', width=1.5),
    mpf.make_addplot(plot_df['upper_threshold'], color='green', linestyle='--', width=1),
    mpf.make_addplot(plot_df['lower_threshold'], color='red', linestyle='--', width=1),
    mpf.make_addplot(plot_df['long_marker'], type='scatter', marker='^', markersize=100, color='lime'),
    mpf.make_addplot(plot_df['short_marker'], type='scatter', marker='v', markersize=100, color='red'),
]

mpf.plot(plot_df, type='candle', style='charles', addplot=add_plots,
         volume=True, figsize=(14, 8), title='NQ with Crossover Signals')


long_entries = df['long_entry'].values
short_entries = df['short_entry'].values

pf = vbt.Portfolio.from_signals(
    close=df['Close'],
    entries=long_entries,
    exits=short_entries,        # Exit long when short signal
    short_entries=short_entries,
    short_exits=long_entries,   # Exit short when long signal
    init_cash=50000,
    size=1,
    size_type='amount',         # 1 contract
    fees=0.0001,
    freq='1h',
    accumulate=False,           # Don't add to position, just flip
)

# Results
print(pf.stats())
pf.plot().show()