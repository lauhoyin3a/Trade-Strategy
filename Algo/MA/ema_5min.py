"""
Find Best 10 EMAs for NAS100 Support/Resistance
Simple analysis - BOUNCE PERCENTAGE ONLY
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


# ============================================================================
# CONFIGURATION
# ============================================================================

CSV_FILE = 'Data/NAS100_0.5years_M5.csv'
EMA_PERIODS = range(15, 30)  # Test EMA 15 to 29
ATR_PERIOD = 14
ATR_MULTIPLIER = 0.6


# ============================================================================
# DATA LOADING
# ============================================================================

def load_data(filepath):
    """Load CSV data"""
    print(f"Loading data from {filepath}...")

    df = pd.read_csv(filepath)
    df.columns = [col.lower().strip() for col in df.columns]

    # Find date column
    date_cols = ['time', 'datetime', 'date', 'timestamp']
    date_col = next((col for col in date_cols if col in df.columns), df.columns[0])

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()

    print(f"✅ Loaded {len(df):,} bars")
    print(f"📅 {df.index[0]} to {df.index[-1]}")
    print(f"💰 Price: ${df['close'].min():.2f} - ${df['close'].max():.2f}\n")

    return df


# ============================================================================
# TECHNICAL INDICATORS
# ============================================================================

def calculate_ema(series, period):
    """Calculate Exponential Moving Average"""
    return series.ewm(span=period, adjust=False).mean()


def calculate_atr(df, period=14):
    """Calculate Average True Range"""
    high = df['high']
    low = df['low']
    close = df['close']

    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


# ============================================================================
# BOUNCE DETECTION
# ============================================================================

def detect_bounces(df, ema_period, atr_period=14, atr_multiplier=0.5):
    """Detect bounces and penetrations at EMA"""
    df = df.copy()

    # Calculate EMA and ATR
    df['ema'] = calculate_ema(df['close'], ema_period)
    df['atr'] = calculate_atr(df, atr_period)

    # Create bands
    df['upper'] = df['ema'] + (atr_multiplier * df['atr'])
    df['lower'] = df['ema'] - (atr_multiplier * df['atr'])

    # Trend: above EMA = uptrend, below = downtrend
    df['trend'] = np.where(df['close'] > df['ema'], 1, -1)
    df['in_bands'] = (df['close'] >= df['lower']) & (df['close'] <= df['upper'])

    # Track interactions
    support_bounces = 0
    support_penetrations = 0
    resistance_bounces = 0
    resistance_penetrations = 0

    in_interaction = False
    interaction_trend = None

    for i in range(1, len(df)):
        price = df['close'].iloc[i]
        trend = df['trend'].iloc[i]
        in_bands = df['in_bands'].iloc[i]

        # Large moves (cross both bands in one bar)
        if not in_interaction:
            prev_price = df['close'].iloc[i-1]
            upper = df['upper'].iloc[i]
            lower = df['lower'].iloc[i]

            if prev_price > upper and price < lower:
                support_penetrations += 1
                continue
            if prev_price < lower and price > upper:
                resistance_penetrations += 1
                continue

        # Enter bands
        if in_bands and not in_interaction:
            in_interaction = True
            interaction_trend = trend
            continue

        # Exit bands
        if not in_bands and in_interaction:
            upper = df['upper'].iloc[i]
            lower = df['lower'].iloc[i]

            if interaction_trend == 1:  # Testing support
                if price > upper:
                    support_bounces += 1
                else:
                    support_penetrations += 1
            elif interaction_trend == -1:  # Testing resistance
                if price < lower:
                    resistance_bounces += 1
                else:
                    resistance_penetrations += 1

            in_interaction = False
            interaction_trend = None

    # Calculate stats
    total_support = support_bounces + support_penetrations
    total_resistance = resistance_bounces + resistance_penetrations
    total = total_support + total_resistance

    support_bounce_pct = (support_bounces / total_support * 100) if total_support > 0 else 0
    resistance_bounce_pct = (resistance_bounces / total_resistance * 100) if total_resistance > 0 else 0
    bounce_pct = ((support_bounces + resistance_bounces) / total * 100) if total > 0 else 0

    return {
        'ema_period': ema_period,
        'support_bounces': support_bounces,
        'support_penetrations': support_penetrations,
        'support_bounce_pct': support_bounce_pct,
        'resistance_bounces': resistance_bounces,
        'resistance_penetrations': resistance_penetrations,
        'resistance_bounce_pct': resistance_bounce_pct,
        'total_interactions': total,
        'bounce_pct': bounce_pct
    }


# ============================================================================
# TEST ALL EMAs
# ============================================================================

def test_all_emas(df, ema_periods, atr_period, atr_multiplier):
    """Test all EMA periods and rank by BOUNCE % ONLY"""
    print(f"Testing {len(list(ema_periods))} EMA periods...")
    print(f"ATR({atr_period}), Multiplier: {atr_multiplier}")
    print(f"Ranking by: BOUNCE PERCENTAGE ONLY\n")

    results = []

    for ema in tqdm(ema_periods, desc="Testing EMAs"):
        res = detect_bounces(df, ema, atr_period, atr_multiplier)

        results.append({
            'EMA': ema,
            'Bounce %': res['bounce_pct'],
            'Support Bounce %': res['support_bounce_pct'],
            'Resistance Bounce %': res['resistance_bounce_pct'],
            'Interactions': res['total_interactions'],
            'Support Bounces': res['support_bounces'],
            'Support Penetrations': res['support_penetrations'],
            'Resistance Bounces': res['resistance_bounces'],
            'Resistance Penetrations': res['resistance_penetrations']
        })

    df_results = pd.DataFrame(results)
    # Sort by BOUNCE % only
    df_results = df_results.sort_values('Bounce %', ascending=False)

    return df_results


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_top_10(results_df):
    """Plot top 10 EMAs by bounce %"""
    top10 = results_df.head(10)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # Plot 1: Bounce Percentage (MAIN METRIC)
    axes[0, 0].barh(top10['EMA'].astype(str), top10['Bounce %'],
                    color='green', edgecolor='black')
    axes[0, 0].axvline(50, color='red', linestyle='--', alpha=0.5, linewidth=2, label='Random (50%)')
    axes[0, 0].set_xlabel('Bounce %', fontsize=12)
    axes[0, 0].set_ylabel('EMA Period', fontsize=12)
    axes[0, 0].set_title('Top 10 EMAs by Bounce % (MAIN RANKING)', fontsize=14, fontweight='bold')
    axes[0, 0].invert_yaxis()
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3, axis='x')

    # Add percentage labels
    for i, (idx, row) in enumerate(top10.iterrows()):
        axes[0, 0].text(row['Bounce %'], i, f" {row['Bounce %']:.1f}%",
                       va='center', fontweight='bold')

    # Plot 2: Support vs Resistance Bounce %
    x_pos = range(len(top10))
    width = 0.35

    axes[0, 1].barh([i - width/2 for i in x_pos], top10['Support Bounce %'],
                    width, label='Support', color='green', edgecolor='black')
    axes[0, 1].barh([i + width/2 for i in x_pos], top10['Resistance Bounce %'],
                    width, label='Resistance', color='red', edgecolor='black')
    axes[0, 1].axvline(50, color='gray', linestyle='--', alpha=0.5)
    axes[0, 1].set_yticks(x_pos)
    axes[0, 1].set_yticklabels(top10['EMA'].astype(str))
    axes[0, 1].set_xlabel('Bounce %', fontsize=12)
    axes[0, 1].set_ylabel('EMA Period', fontsize=12)
    axes[0, 1].set_title('Support vs Resistance Bounce %', fontsize=14, fontweight='bold')
    axes[0, 1].invert_yaxis()
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3, axis='x')

    # Plot 3: Interaction Count (for reference only)
    axes[1, 0].barh(top10['EMA'].astype(str), top10['Interactions'],
                    color='coral', edgecolor='black')
    axes[1, 0].set_xlabel('Number of Interactions', fontsize=12)
    axes[1, 0].set_ylabel('EMA Period', fontsize=12)
    axes[1, 0].set_title('Interaction Count (Reference)', fontsize=14, fontweight='bold')
    axes[1, 0].invert_yaxis()
    axes[1, 0].grid(True, alpha=0.3, axis='x')

    # Plot 4: Bounces vs Penetrations
    support_total = top10['Support Bounces'] + top10['Support Penetrations']
    resistance_total = top10['Resistance Bounces'] + top10['Resistance Penetrations']

    axes[1, 1].barh([i - width/2 for i in x_pos], support_total,
                    width, label='Support (Total)', color='lightgreen', edgecolor='black')
    axes[1, 1].barh([i + width/2 for i in x_pos], resistance_total,
                    width, label='Resistance (Total)', color='lightcoral', edgecolor='black')
    axes[1, 1].set_yticks(x_pos)
    axes[1, 1].set_yticklabels(top10['EMA'].astype(str))
    axes[1, 1].set_xlabel('Count', fontsize=12)
    axes[1, 1].set_ylabel('EMA Period', fontsize=12)
    axes[1, 1].set_title('Support vs Resistance Interactions', fontsize=14, fontweight='bold')
    axes[1, 1].invert_yaxis()
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    return fig


def plot_all_emas_overview(results_df):
    """Plot overview of all EMAs tested"""
    fig, axes = plt.subplots(2, 1, figsize=(16, 9))

    emas = results_df['EMA'].values

    # Plot 1: Bounce % (PRIMARY METRIC)
    axes[0].plot(emas, results_df['Bounce %'], color='steelblue', linewidth=3)
    axes[0].axhline(50, color='red', linestyle='--', alpha=0.5, linewidth=2, label='Random (50%)')
    axes[0].scatter(results_df.head(10)['EMA'], results_df.head(10)['Bounce %'],
                   color='gold', s=150, edgecolor='red', linewidth=2.5,
                   label='Top 10', zorder=5)

    # Mark #1
    best = results_df.iloc[0]
    axes[0].scatter(best['EMA'], best['Bounce %'],
                   color='lime', s=300, edgecolor='darkgreen', linewidth=3,
                   marker='*', label=f"#1: EMA({best['EMA']}) = {best['Bounce %']:.1f}%", zorder=6)

    axes[0].set_xlabel('EMA Period', fontsize=13)
    axes[0].set_ylabel('Bounce %', fontsize=13)
    axes[0].set_title('Bounce Percentage by EMA Period (PRIMARY RANKING)',
                     fontsize=15, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Support vs Resistance Bounce %
    axes[1].plot(emas, results_df['Support Bounce %'], color='green', linewidth=2,
                marker='o', markersize=4, label='Support Bounce %')
    axes[1].plot(emas, results_df['Resistance Bounce %'], color='red', linewidth=2,
                marker='o', markersize=4, label='Resistance Bounce %')
    axes[1].axhline(50, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)
    axes[1].set_xlabel('EMA Period', fontsize=13)
    axes[1].set_ylabel('Bounce %', fontsize=13)
    axes[1].set_title('Support vs Resistance Bounce %', fontsize=15, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_price_with_ema(df, ema_period, atr_period=14, atr_multiplier=0.6, bars=1000):
    """Plot price chart with EMA"""
    df = df.copy()
    df['ema'] = calculate_ema(df['close'], ema_period)
    df['atr'] = calculate_atr(df, atr_period)
    df['upper'] = df['ema'] + (atr_multiplier * df['atr'])
    df['lower'] = df['ema'] - (atr_multiplier * df['atr'])

    plot_df = df.iloc[-bars:]

    fig, ax = plt.subplots(figsize=(18, 8))

    # Price
    ax.plot(plot_df.index, plot_df['close'], color='black', linewidth=1.5,
           alpha=0.8, label='Close')

    # EMA
    ax.plot(plot_df.index, plot_df['ema'], color='blue', linewidth=2.5,
           label=f'EMA({ema_period})')

    # Bands
    ax.plot(plot_df.index, plot_df['upper'], color='gray', linestyle='--',
           alpha=0.5, linewidth=1)
    ax.plot(plot_df.index, plot_df['lower'], color='gray', linestyle='--',
           alpha=0.5, linewidth=1)
    ax.fill_between(plot_df.index, plot_df['upper'], plot_df['lower'],
                    alpha=0.15, color='gray', label=f'ATR({atr_period}) ± {atr_multiplier}x')

    ax.set_xlabel('Date', fontsize=13)
    ax.set_ylabel('Price ($)', fontsize=13)
    ax.set_title(f'NAS100 with EMA({ema_period}) Support/Resistance',
                fontsize=15, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main execution"""
    print("=" * 70)
    print(" FIND BEST 10 EMAs - BY BOUNCE % ONLY")
    print("=" * 70)
    print()

    # Load data
    df = load_data(CSV_FILE)

    # Test all EMAs
    print("=" * 70)
    print("Testing EMAs")
    print("=" * 70)
    results_df = test_all_emas(df, EMA_PERIODS, ATR_PERIOD, ATR_MULTIPLIER)

    # Show top 10
    print("\n" + "=" * 70)
    print("🏆 TOP 10 EMAs (RANKED BY BOUNCE % ONLY)")
    print("=" * 70)
    print()

    top10 = results_df.head(10)
    print(top10[['EMA', 'Bounce %', 'Interactions', 'Support Bounce %', 'Resistance Bounce %']].to_string(index=False))

    # Best EMA
    best = top10.iloc[0]
    print("\n" + "=" * 70)
    print("🥇 #1 BEST EMA (Highest Bounce %)")
    print("=" * 70)
    print(f"  EMA Period:           {int(best['EMA'])}")
    print(f"  Bounce %:             {best['Bounce %']:.2f}%")
    print(f"  Support Bounce %:     {best['Support Bounce %']:.2f}%")
    print(f"  Resistance Bounce %:  {best['Resistance Bounce %']:.2f}%")
    print(f"\n  Total Interactions:      {int(best['Interactions'])}")
    print(f"  Support Bounces:         {int(best['Support Bounces'])}")
    print(f"  Support Penetrations:    {int(best['Support Penetrations'])}")
    print(f"  Resistance Bounces:      {int(best['Resistance Bounces'])}")
    print(f"  Resistance Penetrations: {int(best['Resistance Penetrations'])}")

    # Performance assessment
    if best['Bounce %'] > 55:
        print(f"\n  ✅ EXCELLENT: {best['Bounce %']:.1f}% significantly beats random (50%)")
    elif best['Bounce %'] > 52:
        print(f"\n  ✅ GOOD: {best['Bounce %']:.1f}% beats random")
    else:
        print(f"\n  ⚠️  MARGINAL: {best['Bounce %']:.1f}% only slightly beats random")

    # Visualizations
    print("\n" + "=" * 70)
    print("Creating Visualizations")
    print("=" * 70)

    print("Plot 1: Top 10 comparison...")
    fig1 = plot_top_10(results_df)
    plt.savefig('top_10_emas.png', dpi=300, bbox_inches='tight')
    print("  ✅ Saved: top_10_emas.png")

    print("Plot 2: All EMAs overview...")
    fig2 = plot_all_emas_overview(results_df)
    plt.savefig('all_emas_overview.png', dpi=300, bbox_inches='tight')
    print("  ✅ Saved: all_emas_overview.png")

    print(f"Plot 3: Price chart with EMA({int(best['EMA'])})...")
    fig3 = plot_price_with_ema(df, int(best['EMA']), ATR_PERIOD, ATR_MULTIPLIER)
    plt.savefig('best_ema_chart.png', dpi=300, bbox_inches='tight')
    print("  ✅ Saved: best_ema_chart.png")

    plt.show()

    # Export
    results_df.to_csv('ema_results_full.csv', index=False)
    print("\n  ✅ Saved: ema_results_full.csv")

    print("\n" + "=" * 70)
    print("✅ COMPLETE!")
    print("=" * 70)

    return results_df


if __name__ == "__main__":
    results = main()