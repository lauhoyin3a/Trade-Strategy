"""
Download MNQ/NQ data using MetaTrader 5 - MUCH SIMPLER than IB!
"""

import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta
import pytz


def initialize_mt5(login=None, password=None, server=None):
    """
    Initialize MT5 connection.

    Parameters:
    -----------
    login : int (optional)
        Your MT5 account number
    password : str (optional)
        Your MT5 password
    server : str (optional)
        Your broker's server name

    If no credentials provided, uses the last logged-in account
    """
    # Initialize MT5
    if not mt5.initialize():
        print("❌ MT5 initialization failed")
        print(f"Error: {mt5.last_error()}")
        return False

    # If credentials provided, login
    if login and password and server:
        authorized = mt5.login(login, password, server)
        if not authorized:
            print(f"❌ Login failed. Error: {mt5.last_error()}")
            mt5.shutdown()
            return False
        print(f"✅ Logged in to account: {login}")
    else:
        # Use already logged-in account
        account_info = mt5.account_info()
        if account_info:
            print(f"✅ Using account: {account_info.login}")
        else:
            print("⚠️  No account logged in. Using default connection.")

    # Print MT5 info
    print(f"MT5 version: {mt5.version()}")
    print(f"Terminal info: {mt5.terminal_info().company}")

    return True


def find_nq_symbol():
    """
    Find available NQ/MNQ symbols in your MT5 broker.
    Different brokers use different symbol names.
    """
    print("\n🔍 Searching for NQ/MNQ symbols...")

    # Common symbol variations
    possible_symbols = [
        'NQ',  # Most common
        'MNQ',  # Micro version
        'NQH25',  # With contract month
        'MNQH25',
        'NAS100',  # Some brokers
        'USTEC',  # Some brokers
        'NDX',
        'NQ100',
        'NQ-MAR25',
        'US100',
    ]

    found_symbols = []

    for symbol in possible_symbols:
        # Try to get symbol info
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is not None:
            found_symbols.append({
                'symbol': symbol,
                'description': symbol_info.description,
                'path': symbol_info.path
            })

    if found_symbols:
        print(f"\n✅ Found {len(found_symbols)} matching symbols:")
        for s in found_symbols:
            print(f"   • {s['symbol']:<15} - {s['description']}")
        return found_symbols[0]['symbol']  # Return first match
    else:
        print("\n⚠️  No NQ symbols found. Searching all available symbols...")

        # Get all symbols and search
        all_symbols = mt5.symbols_get()
        nq_related = [s for s in all_symbols if 'NQ' in s.name or 'NAS' in s.name or 'NDX' in s.name]

        if nq_related:
            print(f"\n✅ Found {len(nq_related)} related symbols:")
            for s in nq_related[:10]:  # Show first 10
                print(f"   • {s.name:<20} - {s.description}")
            return nq_related[0].name
        else:
            print("❌ No NQ-related symbols found in your broker")
            return None


def get_mnq_data_mt5(symbol='NQ', timeframe='H1', years=5):
    """
    Download historical data from MT5.

    Parameters:
    -----------
    symbol : str
        Symbol name (e.g., 'NQ', 'MNQ', 'NAS100')
    timeframe : str
        'M1'  = 1 minute
        'M5'  = 5 minutes
        'M15' = 15 minutes
        'M30' = 30 minutes
        'H1'  = 1 hour
        'H4'  = 4 hours
        'D1'  = Daily
        'W1'  = Weekly
    years : int
        How many years back to download

    Returns:
    --------
    pd.DataFrame with OHLCV data
    """
    print(f"\n{'=' * 70}")
    print(f"DOWNLOADING {symbol} DATA FROM MT5")
    print(f"{'=' * 70}")
    print(f"Timeframe: {timeframe}")
    print(f"Period: {years} years")

    # Map timeframe string to MT5 constant
    timeframe_map = {
        'M1': mt5.TIMEFRAME_M1,
        'M5': mt5.TIMEFRAME_M5,
        'M15': mt5.TIMEFRAME_M15,
        'M30': mt5.TIMEFRAME_M30,
        'H1': mt5.TIMEFRAME_H1,
        'H4': mt5.TIMEFRAME_H4,
        'D1': mt5.TIMEFRAME_D1,
        'W1': mt5.TIMEFRAME_W1,
    }

    if timeframe not in timeframe_map:
        print(f"❌ Invalid timeframe. Use: {list(timeframe_map.keys())}")
        return None

    mt5_timeframe = timeframe_map[timeframe]

    # Enable symbol (in case it's hidden)
    if not mt5.symbol_select(symbol, True):
        print(f"❌ Failed to select symbol: {symbol}")
        print(f"Error: {mt5.last_error()}")
        return None

    # Calculate date range
    utc_to = datetime.now(pytz.UTC)
    utc_from = utc_to - timedelta(days=365 * years)

    print(f"\nFrom: {utc_from}")
    print(f"To:   {utc_to}")
    print("\nDownloading... (this is fast!)")

    # Get rates
    rates = mt5.copy_rates_range(symbol, mt5_timeframe, utc_from, utc_to)

    if rates is None or len(rates) == 0:
        print(f"❌ No data received")
        print(f"Error: {mt5.last_error()}")
        return None

    # Convert to DataFrame
    df = pd.DataFrame(rates)

    # Convert time to datetime
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df = df.set_index('time')

    # Rename columns to standard format
    df = df.rename(columns={
        'open': 'open',
        'high': 'high',
        'low': 'low',
        'close': 'close',
        'tick_volume': 'volume',
        'real_volume': 'real_volume',
        'spread': 'spread'
    })

    # Keep only OHLCV columns
    df = df[['open', 'high', 'low', 'close', 'volume']]

    print(f"\n✅ DOWNLOAD COMPLETE!")
    print(f"{'=' * 70}")
    print(f"Total bars: {len(df):,}")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")

    # Calculate actual coverage
    days_covered = (df.index[-1] - df.index[0]).days
    years_covered = days_covered / 365.25
    print(f"Coverage: {years_covered:.1f} years ({days_covered} days)")

    print(f"Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")

    return df


def get_available_symbols_list():
    """
    List all available symbols in your MT5 broker.
    """
    symbols = mt5.symbols_get()

    if symbols is None:
        print("❌ Failed to get symbols")
        return []

    print(f"\n📊 Total available symbols: {len(symbols)}")

    # Organize by category
    categories = {}
    for s in symbols:
        category = s.path.split('\\')[0] if s.path else 'Other'
        if category not in categories:
            categories[category] = []
        categories[category].append(s)

    print(f"\nCategories found: {list(categories.keys())}")

    return symbols


def test_mt5_connection():
    """
    Test MT5 connection and show account details.
    """
    print("=" * 70)
    print("MT5 CONNECTION TEST")
    print("=" * 70)

    if not initialize_mt5():
        return False

    # Account info
    account_info = mt5.account_info()
    if account_info:
        print(f"\n📊 Account Information:")
        print(f"   Login: {account_info.login}")
        print(f"   Server: {account_info.server}")
        print(f"   Company: {account_info.company}")
        print(f"   Currency: {account_info.currency}")
        print(f"   Balance: {account_info.balance}")
        print(f"   Leverage: 1:{account_info.leverage}")

    # Terminal info
    terminal_info = mt5.terminal_info()
    if terminal_info:
        print(f"\n💻 Terminal Information:")
        print(f"   Build: {terminal_info.build}")
        print(f"   Connected: {terminal_info.connected}")
        print(f"   Path: {terminal_info.path}")

    print("\n✅ MT5 connection is working!")
    return True


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # STEP 1: Initialize MT5

    # Option A: Use already logged-in account (easiest)
    if not initialize_mt5():
        print("\n❌ Failed to initialize MT5")
        print("\nMake sure:")
        print("1. MT5 is installed")
        print("2. You're logged into an account in MT5")
        print("3. MT5 is running")
        exit()


    # STEP 2: Test connection
    print("\nSTEP 2: Testing connection...")
    print("-" * 70)
    test_mt5_connection()

    # STEP 3: Find NQ symbol
    print("\nSTEP 3: Finding NQ/MNQ symbol...")
    print("-" * 70)
    symbol = find_nq_symbol()

    if not symbol:
        print("\n⚠️  Automatic symbol detection failed.")
        print("Please enter the symbol name manually.")
        print("Check your MT5 'Market Watch' window for the correct symbol name.")
        symbol = input("\nEnter symbol name (e.g., NQ, MNQ, NAS100): ").strip()

        if not symbol:
            print("❌ No symbol provided. Exiting.")
            mt5.shutdown()
            exit()

    # STEP 4: Download data
    print(f"\nSTEP 4: Downloading data for {symbol}...")

    # Configuration
    YEARS = 0.5
    TIMEFRAME = 'M5'

    df = get_mnq_data_mt5(symbol=symbol, timeframe=TIMEFRAME, years=YEARS)

    if df is not None:
        print("\n✅ SUCCESS!")

        # Show sample data
        print("\nFirst 5 rows:")
        print(df.head())
        print("\nLast 5 rows:")
        print(df.tail())

        # Save data
        filename_csv = f'{symbol}_{YEARS}years_{TIMEFRAME}.csv'
        df.to_csv(filename_csv)

        print(f"\n💾 Data saved to:")
        print(f"   - {filename_csv}")
        print("\n🎉 Data is ready for MA support/resistance analysis!")

    else:
        print("\n❌ Failed to download data")
        print("\nTroubleshooting:")
        print("1. Check if symbol name is correct in Market Watch")
        print("2. Verify you have market data access for this symbol")
        print("3. Try a different timeframe")

    # Cleanup
    mt5.shutdown()
    print("\n✅ Disconnected from MT5")