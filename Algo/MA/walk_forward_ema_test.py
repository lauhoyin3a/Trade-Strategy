"""
Moving Average Support/Resistance Test for NAS100
Simple analysis with fixed ATR parameters (14, 0.5)
Uses reliability scoring (bounce % + interaction count)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


# ============================================================================
# SECTION 1: DATA LOADING
# ============================================================================

def load_data_from_csv(filepath):
    """
    Load data from CSV file
    """
    print(f"Loading data from {filepath}...")

    df = pd.read_csv(filepath)
    df.columns = [col.lower().strip() for col in df.columns]

    # Find date column
    possible_date_cols = ['time', 'datetime', 'date', 'timestamp']
    date_col = None

    for col in possible_date_cols:
        if col in df.columns:
            date_col = col
            break

    if date_col is None:
        date_col = df.columns[0]
        print(f"⚠️  Using '{date_col}' as datetime column")

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col)
    df = df.sort_index()

    print(f"✅ Loaded {len(df):,} bars of data")
    print(f"📅 Date range: {df.index[0]} to {df.index[-1]}")
    print(f"📊 Columns: {df.columns.tolist()}")
    print(f"💰 Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")

    return df


# ============================================================================
# SECTION 2: TECHNICAL INDICATORS
# ============================================================================

def calculate_atr(df, period=14):
    """Calculate Average True Range (ATR)"""
    high = df['high']
    low = df['low']
    close = df['close']

    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()

    return atr


def calculate_sma(series, period):
    """Calculate Simple Moving Average"""
    return series.rolling(window=period).mean()


# ============================================================================
# SECTION 3: RELIABILITY SCORE
# ============================================================================

def calculate_reliability_score(bounce_pct, interactions, min_interactions=20):
    """
    Reliability score = bounce_pct × interaction_weight

    Weight:
    - Below min_interactions: penalty (linear)
    - Above min_interactions: bonus (logarithmic)
    """
    if interactions == 0:
        return 0.0

    if interactions < min_interactions:
        interaction_weight = interactions / min_interactions
    else:
        interaction_weight = 1 + np.log(interactions / min_interactions) / 5

    return bounce_pct * interaction_weight


# ============================================================================
# SECTION 4: BOUNCE/PENETRATION DETECTION
# ============================================================================

def detect_ma_interactions(df, ma_period=72, atr_period=14, atr_multiplier=0.5):
    """Detect bounces and penetrations at MA levels"""
    df = df.copy()
    df['ma'] = calculate_sma(df['close'], ma_period)
    df['atr'] = calculate_atr(df, atr_period)

    # Create bands
    df['upper_band'] = df['ma'] + (atr_multiplier * df['atr'])
    df['lower_band'] = df['ma'] - (atr_multiplier * df['atr'])

    df['trend'] = np.where(df['close'] > df['ma'], 1, -1)
    df['in_bands'] = ((df['close'] >= df['lower_band']) &
                      (df['close'] <= df['upper_band']))

    # Track bounces/penetrations
    support_bounces = 0
    support_penetrations = 0
    resistance_bounces = 0
    resistance_penetrations = 0

    in_interaction = False
    interaction_start_idx = None
    interaction_trend = None

    bounce_indices = []
    penetration_indices = []

    for i in range(1, len(df)):
        current_price = df['close'].iloc[i]
        current_trend = df['trend'].iloc[i]
        in_bands_now = df['in_bands'].iloc[i]

        # Check for large moves
        if not in_interaction:
            prev_price = df['close'].iloc[i - 1]
            upper = df['upper_band'].iloc[i]
            lower = df['lower_band'].iloc[i]

            if prev_price > upper and current_price < lower:
                support_penetrations += 1
                penetration_indices.append(i)
                continue

            if prev_price < lower and current_price > upper:
                resistance_penetrations += 1
                penetration_indices.append(i)
                continue

        # Enter bands
        if in_bands_now and not in_interaction:
            in_interaction = True
            interaction_start_idx = i
            interaction_trend = current_trend
            continue

        # Exit bands
        if not in_bands_now and in_interaction:
            upper = df['upper_band'].iloc[i]
            lower = df['lower_band'].iloc[i]

            if interaction_trend == 1:
                if current_price > upper:
                    support_bounces += 1
                    bounce_indices.append(i)
                else:
                    support_penetrations += 1
                    penetration_indices.append(i)

            elif interaction_trend == -1:
                if current_price < lower:
                    resistance_bounces += 1
                    bounce_indices.append(i)
                else:
                    resistance_penetrations += 1
                    penetration_indices.append(i)

            in_interaction = False
            interaction_start_idx = None
            interaction_trend = None

    # Calculate statistics
    total_support = support_bounces + support_penetrations
    total_resistance = resistance_bounces + resistance_penetrations
    total_all = total_support + total_resistance

    support_bounce_pct = (support_bounces / total_support * 100) if total_support > 0 else 0
    resistance_bounce_pct = (resistance_bounces / total_resistance * 100) if total_resistance > 0 else 0
    combined_bounce_pct = ((support_bounces + resistance_bounces) / total_all * 100) if total_all > 0 else 0

    return {
        'ma_period': ma_period,
        'support_bounces': support_bounces,
        'support_penetrations': support_penetrations,
        'resistance_bounces': resistance_bounces,
        'resistance_penetrations': resistance_penetrations,
        'support_bounce_pct': support_bounce_pct,
        'resistance_bounce_pct': resistance_bounce_pct,
        'combined_bounce_pct': combined_bounce_pct,
        'total_interactions': total_all,
        'df': df,
        'bounce_indices': bounce_indices,
        'penetration_indices': penetration_indices
    }


# ============================================================================
# SECTION 5: TEST MA RANGE
# ============================================================================

def test_ma_range(df, ma_periods=range(24, 201), atr_period=14, atr_multiplier=0.5,
                  min_interactions=20):
    """Test multiple MA periods with reliability scoring"""
    print(f"\nTesting {len(list(ma_periods))} moving average periods...")
    print(f"ATR Period: {atr_period}, ATR Multiplier: {atr_multiplier}")
    print(f"Minimum interactions threshold: {min_interactions}\n")

    results_list = []

    for ma_period in tqdm(ma_periods):
        result = detect_ma_interactions(df, ma_period, atr_period, atr_multiplier)

        reliability_score = calculate_reliability_score(
            result['combined_bounce_pct'],
            result['total_interactions'],
            min_interactions
        )

        results_list.append({
            'ma_period': ma_period,
            'support_bounces': result['support_bounces'],
            'support_penetrations': result['support_penetrations'],
            'support_bounce_pct': result['support_bounce_pct'],
            'resistance_bounces': result['resistance_bounces'],
            'resistance_penetrations': result['resistance_penetrations'],
            'resistance_bounce_pct': result['resistance_bounce_pct'],
            'combined_bounce_pct': result['combined_bounce_pct'],
            'total_interactions': result['total_interactions'],
            'reliability_score': reliability_score
        })

    results_df = pd.DataFrame(results_list)

    # Find best by reliability score
    best_idx = results_df['reliability_score'].idxmax()
    best_row = results_df.loc[best_idx]

    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"\n🏆 BEST MA PERIOD (by Reliability Score): {int(best_row['ma_period'])}")
    print(f"   Bounce %:        {best_row['combined_bounce_pct']:.2f}%")
    print(f"   Interactions:    {int(best_row['total_interactions'])}")
    print(f"   Reliability:     {best_row['reliability_score']:.2f}")

    print(f"\n📊 Breakdown:")
    print(f"   Support bounces:     {int(best_row['support_bounces'])} ({best_row['support_bounce_pct']:.1f}%)")
    print(f"   Support penetrations: {int(best_row['support_penetrations'])}")
    print(f"   Resistance bounces:   {int(best_row['resistance_bounces'])} ({best_row['resistance_bounce_pct']:.1f}%)")
    print(f"   Resistance penetrations: {int(best_row['resistance_penetrations'])}")

    if best_row['combined_bounce_pct'] > 55:
        print(f"\n✅ Strong performance! Bounce rate significantly beats random (50%)")
    elif best_row['combined_bounce_pct'] > 52:
        print(f"\n✅ Moderate performance, beats random")
    else:
        print(f"\n⚠️  Marginal performance")

    return results_df


# ============================================================================
# SECTION 6: VISUALIZATION
# ============================================================================

def plot_ma_comparison(results_df):
    """Plot comparison of different MA periods"""
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))

    ma_vals = results_df['ma_period'].values

    # Plot 1: Bounce Percentage
    axes[0].plot(ma_vals, results_df['combined_bounce_pct'],
                 color='steelblue', linewidth=2, marker='o', markersize=4)
    axes[0].axhline(50, color='red', linestyle='--', alpha=0.5, label='Random (50%)')

    best_idx = results_df['reliability_score'].idxmax()
    best_ma = results_df.loc[best_idx, 'ma_period']
    best_bounce = results_df.loc[best_idx, 'combined_bounce_pct']
    axes[0].plot(best_ma, best_bounce, 'r*', markersize=20,
                 label=f'Best: MA({int(best_ma)}) = {best_bounce:.1f}%')

    axes[0].set_xlabel('MA Period', fontsize=12)
    axes[0].set_ylabel('Bounce %', fontsize=12)
    axes[0].set_title('Bounce Percentage by MA Period', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Interaction Count
    axes[1].plot(ma_vals, results_df['total_interactions'],
                 color='coral', linewidth=2, marker='o', markersize=4)
    axes[1].set_xlabel('MA Period', fontsize=12)
    axes[1].set_ylabel('Number of Interactions', fontsize=12)
    axes[1].set_title('MA Interaction Count (Touches)', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Reliability Score
    axes[2].plot(ma_vals, results_df['reliability_score'],
                 color='green', linewidth=2, marker='o', markersize=4)

    best_reliability = results_df.loc[best_idx, 'reliability_score']
    axes[2].plot(best_ma, best_reliability, 'r*', markersize=20,
                 label=f'Best: MA({int(best_ma)}) = {best_reliability:.1f}')

    axes[2].set_xlabel('MA Period', fontsize=12)
    axes[2].set_ylabel('Reliability Score', fontsize=12)
    axes[2].set_title('Reliability Score (Bounce % × Interaction Weight)', fontsize=14, fontweight='bold')
    axes[2].legend(fontsize=11)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_ma_with_bounces(df, ma_period=72, atr_period=14, atr_multiplier=0.5, sample_bars=1000):
    """Visualize price with MA and bounces/penetrations"""
    result = detect_ma_interactions(df, ma_period, atr_period, atr_multiplier)
    plot_df = result['df'].iloc[-sample_bars:]

    fig, ax = plt.subplots(figsize=(18, 8))

    # Plot price
    ax.plot(plot_df.index, plot_df['close'], label='Close Price',
            color='black', linewidth=1.5, alpha=0.8)

    # Plot MA
    ax.plot(plot_df.index, plot_df['ma'], label=f'MA({ma_period})',
            color='blue', linewidth=2.5)

    # Plot bands
    ax.plot(plot_df.index, plot_df['upper_band'],
            color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.plot(plot_df.index, plot_df['lower_band'],
            color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.fill_between(plot_df.index, plot_df['upper_band'], plot_df['lower_band'],
                    alpha=0.15, color='gray', label=f'ATR({atr_period}) ± {atr_multiplier}x')

    # Mark bounces and penetrations
    bounce_count = 0
    penetration_count = 0

    for idx in result['bounce_indices']:
        if idx >= len(df) - sample_bars:
            ax.axvline(df.index[idx], color='green', alpha=0.25, linewidth=1.5)
            bounce_count += 1

    for idx in result['penetration_indices']:
        if idx >= len(df) - sample_bars:
            ax.axvline(df.index[idx], color='red', alpha=0.25, linewidth=1.5)
            penetration_count += 1

    # Add custom legend entries
    from matplotlib.lines import Line2D
    custom_lines = [
        Line2D([0], [0], color='green', alpha=0.5, linewidth=3, label=f'Bounces ({bounce_count})'),
        Line2D([0], [0], color='red', alpha=0.5, linewidth=3, label=f'Penetrations ({penetration_count})')
    ]

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles + custom_lines, labels + [l.get_label() for l in custom_lines],
             loc='best', fontsize=11)

    ax.set_xlabel('Date', fontsize=13)
    ax.set_ylabel('Price ($)', fontsize=13)
    ax.set_title(f'NAS100 with MA({ma_period}) Support/Resistance | ATR({atr_period}, {atr_multiplier}x)\n'
                 f'Total Interactions: {result["total_interactions"]} | '
                 f'Bounce Rate: {result["combined_bounce_pct"]:.1f}%',
                 fontsize=15, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_detailed_breakdown(results_df):
    """Create detailed breakdown showing support vs resistance"""
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))

    ma_vals = results_df['ma_period'].values

    # Plot 1: Support vs Resistance Bounce %
    axes[0].plot(ma_vals, results_df['support_bounce_pct'],
                 color='green', linewidth=2, marker='o', markersize=3, label='Support Bounce %')
    axes[0].plot(ma_vals, results_df['resistance_bounce_pct'],
                 color='red', linewidth=2, marker='o', markersize=3, label='Resistance Bounce %')
    axes[0].axhline(50, color='gray', linestyle='--', alpha=0.5, label='Random (50%)')

    axes[0].set_xlabel('MA Period', fontsize=12)
    axes[0].set_ylabel('Bounce %', fontsize=12)
    axes[0].set_title('Support vs Resistance Bounce Percentage', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Support vs Resistance Interaction Count
    axes[1].plot(ma_vals, results_df['support_bounces'] + results_df['support_penetrations'],
                 color='green', linewidth=2, marker='o', markersize=3, label='Support Interactions')
    axes[1].plot(ma_vals, results_df['resistance_bounces'] + results_df['resistance_penetrations'],
                 color='red', linewidth=2, marker='o', markersize=3, label='Resistance Interactions')

    axes[1].set_xlabel('MA Period', fontsize=12)
    axes[1].set_ylabel('Number of Interactions', fontsize=12)
    axes[1].set_title('Support vs Resistance Interaction Count', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


# ============================================================================
# SECTION 7: MAIN EXECUTION
# ============================================================================

def main(csv_file='Data/NAS100_5years_H1.csv'):
    """
    Main execution - simple MA analysis with fixed ATR parameters
    """
    print("=" * 70)
    print(" MA SUPPORT/RESISTANCE TEST - NAS100")
    print("=" * 70)
    print("\n Configuration:")
    print("   ATR Period: 14 (standard)")
    print("   ATR Multiplier: 0.5")
    print("   MA Range: 24-200")

    # Configuration
    MA_PERIODS = range(15, 201)
    ATR_PERIOD = 14
    ATR_MULTIPLIER = 0.5
    MIN_INTERACTIONS = 20

    # Load data
    print("\n" + "=" * 70)
    print("Loading Data")
    print("=" * 70)

    try:
        df = load_data_from_csv(csv_file)
    except Exception as e:
        print(f"\n❌ Failed to load data: {e}")
        return None

    # Test MA range
    print("\n" + "=" * 70)
    print("Testing MA Periods")
    print("=" * 70)

    results_df = test_ma_range(df, MA_PERIODS, ATR_PERIOD, ATR_MULTIPLIER, MIN_INTERACTIONS)

    # Get best MA
    best_idx = results_df['reliability_score'].idxmax()
    best_ma = int(results_df.loc[best_idx, 'ma_period'])

    # Visualizations
    print("\n" + "=" * 70)
    print("Creating Visualizations")
    print("=" * 70)

    print("Creating plot 1: MA comparison...")
    fig1 = plot_ma_comparison(results_df)
    plt.savefig('ma_comparison.png', dpi=300, bbox_inches='tight')
    print("  ✅ Saved: ma_comparison.png")

    print("Creating plot 2: Detailed breakdown...")
    fig2 = plot_detailed_breakdown(results_df)
    plt.savefig('ma_breakdown.png', dpi=300, bbox_inches='tight')
    print("  ✅ Saved: ma_breakdown.png")

    print(f"Creating plot 3: Price chart with MA({best_ma})...")
    fig3 = plot_ma_with_bounces(df, ma_period=best_ma, atr_period=ATR_PERIOD,
                                atr_multiplier=ATR_MULTIPLIER, sample_bars=1000)
    plt.savefig('ma_price_chart.png', dpi=300, bbox_inches='tight')
    print("  ✅ Saved: ma_price_chart.png")

    plt.show()

    # Export results
    results_df.to_csv('ma_test_results.csv', index=False)
    print("\n  ✅ Saved: ma_test_results.csv")

    print("\n" + "=" * 70)
    print("✅ Analysis Complete!")
    print("=" * 70)

    return {
        'results_df': results_df,
        'best_ma': best_ma
    }


if __name__ == "__main__":

    CSV_FILE = 'Data/NAS100_5years_H1.csv'

    print(f"\n🎯 Data file: {CSV_FILE}")
    print()

    analysis_results = main(csv_file=CSV_FILE)

    if analysis_results is not None:
        print("\nGenerated files:")
        print("  📊 ma_comparison.png - Performance metrics by MA period")
        print("  📊 ma_breakdown.png - Support vs Resistance breakdown")
        print("  📊 ma_price_chart.png - Visual with bounces/penetrations")
        print("  📄 ma_test_results.csv - Full results table")