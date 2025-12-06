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


# assume now is bar open
def calculate_thresholds(df, vol_lookback=20, chart_res_minutes=60):
    # for 1-hour data
    # Trading hours per day: 6.5 hours (390 minutes)
    # Trading days per year: 252
    # Hours per year: 252 * 6.5 = 1638 hours
    periods_per_year = (252.0 * 6.5)
    # periods_per_year = (252.0 * 390.0) / chart_res_minutes


    # curr bar close return dont use for calculation!! look ahead bias
    df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
    # Calculate annualized volatility
    # Standard deviation of log returns * sqrt(periods per year)
    df['volatility'] = df['log_return'].rolling(window=vol_lookback).std() * np.sqrt(periods_per_year)

    # Volatility at bar N: uses returns through bar N
    prev_close = df['Close'].shift(1)
    prev_vol = df['volatility'].shift(1)  # Vol calculated through bar N-1

    df['expected_move'] = prev_close * prev_vol * np.sqrt(1.0 / periods_per_year)
    df['upper_threshold'] = prev_close + df['expected_move']
    df['lower_threshold'] = prev_close - df['expected_move']

    return df


def detect_crossover_signals(df):
    """
    At bar N's open:
    - Check if bar N-1's close crossed the threshold that was active during bar N-1
    - Threshold active during bar N-1 = threshold[N-1] (based on Close[N-2])
    """
    # Bar N-2's close (before the crossover)
    close_before = df['Close'].shift(2)

    # Bar N-1's close (after the crossover)
    close_after = df['Close'].shift(1)

    # Threshold that was active during bar N-1 (based on Close[N-2])
    threshold_upper = df['upper_threshold'].shift(1)
    threshold_lower = df['lower_threshold'].shift(1)
    before_threshold_upper = df['upper_threshold'].shift(2)
    before_threshold_lower = df['lower_threshold'].shift(2)

    # MA filter (use bar N-1's values)
    price_above_sma = close_after > df['sma20'].shift(1)
    price_below_sma = close_after < df['sma20'].shift(1)

    # Crossover: was below/at threshold, now above/below
    df['long_signal'] = (close_before <= before_threshold_upper) & (close_after > threshold_upper) & price_above_sma
    df['short_signal'] = (close_before >= before_threshold_lower) & (close_after < threshold_lower) & price_below_sma

    # Entry at bar N's open
    df['long_entry'] = df['long_signal']
    df['short_entry'] = df['short_signal']

    return df

def plot_trade(df):
    df.index = pd.to_datetime(df.index, utc=True).tz_convert(None)
    df['long_marker'] = np.where(df['long_entry'], df['Low'] * 0.999, np.nan)
    df['short_marker'] = np.where(df['short_entry'], df['High'] * 1.001, np.nan)
    plot_df = df.tail(200).copy()

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




# Get and save 1-hour data
# data = get_nq_1h()
# data.to_csv("NQ_1h.csv")

df = pd.read_csv('NQ_1h.csv', index_col=0, parse_dates=True)
df['sma20'] = df['Close'].rolling(20).mean()
df = calculate_thresholds(df, 70)
df = detect_crossover_signals(df)

plot_trade(df)
# user 2024
# df = df[df.index.year == 2025]

long_entries = df['long_entry'].values
short_entries = df['short_entry'].values
# # Stop loss = 1x expected move as a % of current price
# df['sl_pct'] = df['expected_move'] * 0.7 / df['Close'].shift(1)
#
# # Take profit = 3x expected move as a % of current price
# df['tp_pct'] = (df['expected_move'] * 2) / df['Close'].shift(1)

trail_multiplier = 1.5
df['ts_pct'] = (df['expected_move'] * trail_multiplier) / df['Close'].shift(1)



pf = vbt.Portfolio.from_signals(
    close=df['Close'],  # For portfolio valuation
    open=df['Open'],  # For entry/exit execution
    high=df['High'],  # For stop loss checks (intrabar)
    low=df['Low'],  # For stop loss checks (intrabar)
    entries=long_entries,
    exits=short_entries,        # Exit long when short signal
    short_entries=short_entries,
    short_exits=long_entries,   # Exit short when long signal
    init_cash=50000,
    size=1,
    size_type='amount',         # 1 contract
    fees=0,
    fixed_fees=2.50,
    slippage=0.00005,
    freq='1h',
    accumulate=True,  # Enable pyramiding
    max_size=3,  # Max 5 contracts (like pyramiding=5)
    size_granularity=1,  # Trade in whole contracts
    sl_stop=df['ts_pct'].values,
    price=df['Open'],  # Execute trades at Open price
    stop_exit_price='stoplimit',  # Use exact stop price, not close
)

# Results
print(pf.stats())
pf.plot().show()