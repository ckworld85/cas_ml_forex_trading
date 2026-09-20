"""
Forex Technical Fingerprint Script

This script calculates a "technical fingerprint" to characterize forex market conditions
for both the current state and historical periods. It uses rolling windows across all
available historical data to identify periods with similar market characteristics.

Fingerprint Components (15 Features):
1. Volatility (4 features): Avg Daily Range, Avg ATR-14, Avg BB Width, Intraday Range % of ATR
2. Trend (6 features): Avg/Max Consecutive Up/Down Days, % Days Up/Down
3. Momentum (3 features): % RSI overbought/oversold/neutral
4. Volume (1 feature): Avg Daily Volume
5. Mean Reversion (1 feature): Avg Absolute Distance from SMA-50

Usage:
    python fingerprint.py                           # Run with default settings
    python fingerprint.py --window-size 45          # Custom window size
    python fingerprint.py --window-step 21          # Custom step size
    python fingerprint.py --output-dir ./output     # Custom output directory
    python fingerprint.py --parallel-jobs 4         # Enable parallel processing
    python fingerprint.py --export-csv              # Export results to CSV
    python fingerprint.py --export-json             # Export results to JSON
"""

import os
import sys
import json
import argparse
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import ta

# Add project root to Python path to enable ModelTrading package imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import ModelTrading.config.directories as dir_config
import ModelTrading.source.python.utils.csv as csv_utils
from .fingerprint_config import get_fingerprint_config


@dataclass
class Fingerprint:
    """
    Data class representing a technical fingerprint for a time window.
    
    Contains all 15 fingerprint features plus metadata about the window.
    """
    # Window metadata
    window_start: str
    window_end: str
    window_size_days: int
    is_current: bool = False
    
    # Volatility features (4)
    avg_daily_range: float = 0.0
    avg_atr_14: float = 0.0
    avg_bb_width: float = 0.0
    intraday_range_pct_atr: float = 0.0
    
    # Trend features (6)
    avg_consecutive_up_days: float = 0.0
    avg_consecutive_down_days: float = 0.0
    max_consecutive_up_days: int = 0
    max_consecutive_down_days: int = 0
    pct_days_up: float = 0.0
    pct_days_down: float = 0.0
    
    # Momentum features (3)
    pct_rsi_overbought: float = 0.0
    pct_rsi_oversold: float = 0.0
    pct_rsi_neutral: float = 0.0
    
    # Volume features (1)
    avg_daily_volume: float = 0.0
    
    # Mean reversion features (1)
    avg_abs_distance_sma_50: float = 0.0
    
    def to_feature_vector(self) -> np.ndarray:
        """Convert fingerprint to a numpy array of feature values for comparison"""
        return np.array([
            self.avg_daily_range,
            self.avg_atr_14,
            self.avg_bb_width,
            self.intraday_range_pct_atr,
            self.avg_consecutive_up_days,
            self.avg_consecutive_down_days,
            self.max_consecutive_up_days,
            self.max_consecutive_down_days,
            self.pct_days_up,
            self.pct_days_down,
            self.pct_rsi_overbought,
            self.pct_rsi_oversold,
            self.pct_rsi_neutral,
            self.avg_daily_volume,
            self.avg_abs_distance_sma_50
        ], dtype=np.float64)
    
    @staticmethod
    def feature_names() -> List[str]:
        """Returns list of feature names in the same order as to_feature_vector"""
        return [
            'avg_daily_range',
            'avg_atr_14',
            'avg_bb_width',
            'intraday_range_pct_atr',
            'avg_consecutive_up_days',
            'avg_consecutive_down_days',
            'max_consecutive_up_days',
            'max_consecutive_down_days',
            'pct_days_up',
            'pct_days_down',
            'pct_rsi_overbought',
            'pct_rsi_oversold',
            'pct_rsi_neutral',
            'avg_daily_volume',
            'avg_abs_distance_sma_50'
        ]


@dataclass
class SimilarityResult:
    """Result of comparing a historical fingerprint to the current one"""
    fingerprint: Fingerprint
    distance: float
    rank: int = 0


def calculate_indicators(df: pd.DataFrame, params: Dict) -> pd.DataFrame:
    """
    Calculate all required technical indicators for fingerprint analysis.
    
    Args:
        df: DataFrame with OHLCV data (must have columns: open, high, low, close, volume)
        params: Indicator parameters from config
        
    Returns:
        DataFrame with added indicator columns
    """
    df = df.copy()
    
    # ATR (Average True Range)
    atr_indicator = ta.volatility.AverageTrueRange(
        high=df['high'], 
        low=df['low'], 
        close=df['close'], 
        window=params['atr_period']
    )
    df['atr'] = atr_indicator.average_true_range()
    
    # Bollinger Bands
    bb = ta.volatility.BollingerBands(
        close=df['close'], 
        window=params['bb_period'], 
        window_dev=params['bb_std']
    )
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_lower'] = bb.bollinger_lband()
    df['bb_width'] = bb.bollinger_wband()
    
    # RSI
    rsi_indicator = ta.momentum.RSIIndicator(
        close=df['close'], 
        window=params['rsi_period']
    )
    df['rsi'] = rsi_indicator.rsi()
    
    # SMA for mean reversion
    sma_indicator = ta.trend.SMAIndicator(
        close=df['close'], 
        window=params['sma_period']
    )
    df['sma_50'] = sma_indicator.sma_indicator()
    
    # Daily range (high - low)
    df['daily_range'] = df['high'] - df['low']
    
    # Daily direction (1 = up, -1 = down, 0 = unchanged)
    df['daily_direction'] = np.sign(df['close'] - df['open'])
    
    # Intraday range as percentage of ATR
    df['intraday_range_pct_atr'] = np.where(
        df['atr'] > 0,
        (df['daily_range'] / df['atr']) * 100,
        0
    )
    
    # Distance from SMA-50 (absolute value, as percentage of price)
    df['abs_distance_sma_50'] = np.abs(df['close'] - df['sma_50']) / df['close'] * 100
    
    return df


def calculate_consecutive_runs(directions: np.ndarray) -> Tuple[List[int], List[int]]:
    """
    Calculate consecutive runs of up and down days.
    
    Args:
        directions: Array of daily directions (1 = up, -1 = down, 0 = unchanged)
        
    Returns:
        Tuple of (up_runs, down_runs) where each is a list of run lengths
    """
    up_runs = []
    down_runs = []
    
    current_up_run = 0
    current_down_run = 0
    
    for direction in directions:
        if direction > 0:
            current_up_run += 1
            if current_down_run > 0:
                down_runs.append(current_down_run)
                current_down_run = 0
        elif direction < 0:
            current_down_run += 1
            if current_up_run > 0:
                up_runs.append(current_up_run)
                current_up_run = 0
        else:
            # Unchanged - end both runs
            if current_up_run > 0:
                up_runs.append(current_up_run)
                current_up_run = 0
            if current_down_run > 0:
                down_runs.append(current_down_run)
                current_down_run = 0
    
    # Don't forget the last run
    if current_up_run > 0:
        up_runs.append(current_up_run)
    if current_down_run > 0:
        down_runs.append(current_down_run)
    
    return up_runs, down_runs


def calculate_fingerprint(df_window: pd.DataFrame, window_start: str, window_end: str, 
                          rsi_thresholds: Dict, is_current: bool = False) -> Fingerprint:
    """
    Calculate a fingerprint for a single time window.
    
    Args:
        df_window: DataFrame slice for the window (must have indicator columns)
        window_start: Start date string
        window_end: End date string
        rsi_thresholds: RSI threshold configuration
        is_current: Whether this is the current (most recent) fingerprint
        
    Returns:
        Fingerprint object with all calculated features
    """
    n_days = len(df_window)
    
    # Volatility features
    avg_daily_range = df_window['daily_range'].mean()
    avg_atr_14 = df_window['atr'].mean()
    avg_bb_width = df_window['bb_width'].mean()
    intraday_range_pct_atr = df_window['intraday_range_pct_atr'].mean()
    
    # Trend features
    directions = df_window['daily_direction'].values
    up_runs, down_runs = calculate_consecutive_runs(directions)
    
    avg_consecutive_up_days = np.mean(up_runs) if up_runs else 0.0
    avg_consecutive_down_days = np.mean(down_runs) if down_runs else 0.0
    max_consecutive_up_days = max(up_runs) if up_runs else 0
    max_consecutive_down_days = max(down_runs) if down_runs else 0
    
    pct_days_up = (directions > 0).sum() / n_days * 100 if n_days > 0 else 0.0
    pct_days_down = (directions < 0).sum() / n_days * 100 if n_days > 0 else 0.0
    
    # Momentum features (RSI thresholds)
    rsi_values = df_window['rsi'].dropna()
    n_rsi = len(rsi_values)
    
    if n_rsi > 0:
        pct_rsi_overbought = (rsi_values > rsi_thresholds['overbought']).sum() / n_rsi * 100
        pct_rsi_oversold = (rsi_values < rsi_thresholds['oversold']).sum() / n_rsi * 100
        pct_rsi_neutral = (
            (rsi_values >= rsi_thresholds['neutral_low']) & 
            (rsi_values <= rsi_thresholds['neutral_high'])
        ).sum() / n_rsi * 100
    else:
        pct_rsi_overbought = 0.0
        pct_rsi_oversold = 0.0
        pct_rsi_neutral = 0.0
    
    # Volume features
    avg_daily_volume = df_window['volume'].mean()
    
    # Mean reversion features
    avg_abs_distance_sma_50 = df_window['abs_distance_sma_50'].mean()
    
    return Fingerprint(
        window_start=window_start,
        window_end=window_end,
        window_size_days=n_days,
        is_current=is_current,
        avg_daily_range=float(avg_daily_range),
        avg_atr_14=float(avg_atr_14),
        avg_bb_width=float(avg_bb_width),
        intraday_range_pct_atr=float(intraday_range_pct_atr),
        avg_consecutive_up_days=float(avg_consecutive_up_days),
        avg_consecutive_down_days=float(avg_consecutive_down_days),
        max_consecutive_up_days=int(max_consecutive_up_days),
        max_consecutive_down_days=int(max_consecutive_down_days),
        pct_days_up=float(pct_days_up),
        pct_days_down=float(pct_days_down),
        pct_rsi_overbought=float(pct_rsi_overbought),
        pct_rsi_oversold=float(pct_rsi_oversold),
        pct_rsi_neutral=float(pct_rsi_neutral),
        avg_daily_volume=float(avg_daily_volume),
        avg_abs_distance_sma_50=float(avg_abs_distance_sma_50)
    )


def _process_window_for_parallel(window_data):
    """
    Helper function for parallel processing of window fingerprints.
    Must be a top-level function to be picklable.
    
    Args:
        window_data: Tuple of (start_idx, end_idx, window_start, window_end, df_slice_dict, rsi_thresh)
        
    Returns:
        Fingerprint object for the window
    """
    start_idx, end_idx, window_start, window_end, df_slice_dict, rsi_thresh = window_data
    df_slice = pd.DataFrame(df_slice_dict)
    return calculate_fingerprint(df_slice, window_start, window_end, rsi_thresh)


def calculate_historical_fingerprints(df: pd.DataFrame, window_size: int, window_step: int,
                                       rsi_thresholds: Dict, parallel_jobs: int = 1) -> List[Fingerprint]:
    """
    Calculate fingerprints for all historical rolling windows.
    
    Args:
        df: DataFrame with OHLCV data and calculated indicators
        window_size: Rolling window size in days
        window_step: Step size between windows in days
        rsi_thresholds: RSI threshold configuration
        parallel_jobs: Number of parallel jobs (1 = sequential)
        
    Returns:
        List of Fingerprint objects for all historical windows
    """
    n = len(df)
    fingerprints = []
    
    # Generate window start/end positions
    # Reserve the last window_size days for the current fingerprint
    max_start = n - window_size - 1  # -1 to ensure at least 1 day gap from current
    
    windows = []
    start_idx = 0
    while start_idx <= max_start:
        end_idx = start_idx + window_size
        window_start = df.index[start_idx].strftime('%Y-%m-%d')
        window_end = df.index[end_idx - 1].strftime('%Y-%m-%d')
        windows.append((start_idx, end_idx, window_start, window_end))
        start_idx += window_step
    
    print(f"Calculating {len(windows)} historical fingerprints...")
    
    if parallel_jobs > 1 and len(windows) > parallel_jobs:
        # Parallel processing
        print(f"  Using {parallel_jobs} parallel workers")
        
        with ProcessPoolExecutor(max_workers=parallel_jobs) as executor:
            futures = []
            for start_idx, end_idx, window_start, window_end in windows:
                df_slice = df.iloc[start_idx:end_idx]
                # Convert to dict for serialization
                window_data = (
                    start_idx, end_idx, window_start, window_end,
                    df_slice.to_dict(orient='list'),
                    rsi_thresholds
                )
                futures.append(executor.submit(_process_window_for_parallel, window_data))
            
            for future in as_completed(futures):
                fingerprints.append(future.result())
    else:
        # Sequential processing
        for start_idx, end_idx, window_start, window_end in windows:
            df_slice = df.iloc[start_idx:end_idx]
            fp = calculate_fingerprint(df_slice, window_start, window_end, rsi_thresholds)
            fingerprints.append(fp)
    
    # Sort by window start date
    fingerprints.sort(key=lambda x: x.window_start)
    
    return fingerprints


def calculate_current_fingerprint(df: pd.DataFrame, window_size: int, 
                                   rsi_thresholds: Dict) -> Fingerprint:
    """
    Calculate the current (most recent) fingerprint.
    
    Args:
        df: DataFrame with OHLCV data and calculated indicators
        window_size: Window size in days
        rsi_thresholds: RSI threshold configuration
        
    Returns:
        Fingerprint for the most recent window
    """
    # Use the last window_size days
    df_current = df.iloc[-window_size:]
    window_start = df_current.index[0].strftime('%Y-%m-%d')
    window_end = df_current.index[-1].strftime('%Y-%m-%d')
    
    return calculate_fingerprint(df_current, window_start, window_end, rsi_thresholds, is_current=True)


def normalize_fingerprints(current: Fingerprint, historical: List[Fingerprint]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Normalize fingerprint features using z-score normalization based on historical data.
    
    This ensures that each feature contributes equally to distance calculations.
    
    Args:
        current: Current fingerprint
        historical: List of historical fingerprints
        
    Returns:
        Tuple of (normalized_current_vector, normalized_historical_matrix)
    """
    # Stack all historical fingerprints into a matrix
    historical_matrix = np.vstack([fp.to_feature_vector() for fp in historical])
    current_vector = current.to_feature_vector()
    
    # Calculate mean and std from historical data
    means = historical_matrix.mean(axis=0)
    stds = historical_matrix.std(axis=0)
    
    # Avoid division by zero
    stds = np.where(stds == 0, 1, stds)
    
    # Normalize
    normalized_historical = (historical_matrix - means) / stds
    normalized_current = (current_vector - means) / stds
    
    return normalized_current, normalized_historical


def find_similar_windows(current: Fingerprint, historical: List[Fingerprint], 
                         top_n: int = 10) -> List[SimilarityResult]:
    """
    Find the most similar historical windows to the current fingerprint.
    
    Uses Euclidean distance on normalized feature vectors.
    
    Args:
        current: Current fingerprint
        historical: List of historical fingerprints
        top_n: Number of top similar windows to return
        
    Returns:
        List of SimilarityResult objects, sorted by distance (ascending)
    """
    if not historical:
        return []
    
    # Normalize features
    norm_current, norm_historical = normalize_fingerprints(current, historical)
    
    # Calculate Euclidean distances
    distances = np.sqrt(np.sum((norm_historical - norm_current) ** 2, axis=1))
    
    # Create results and sort by distance
    results = []
    for i, (fp, dist) in enumerate(zip(historical, distances)):
        results.append(SimilarityResult(fingerprint=fp, distance=float(dist)))
    
    results.sort(key=lambda x: x.distance)
    
    # Assign ranks and return top N
    for i, result in enumerate(results[:top_n]):
        result.rank = i + 1
    
    return results[:top_n]


def export_to_csv(current: Fingerprint, historical: List[Fingerprint], 
                  similar: List[SimilarityResult], output_dir: str):
    """
    Export fingerprints to CSV files.
    
    Args:
        current: Current fingerprint
        historical: List of all historical fingerprints
        similar: List of most similar windows
        output_dir: Output directory path
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Current fingerprint
    current_df = pd.DataFrame([asdict(current)])
    current_path = os.path.join(output_dir, "fingerprint_current.csv")
    current_df.to_csv(current_path, index=False)
    print(f"  Current fingerprint saved to: {current_path}")
    
    # All historical fingerprints
    historical_df = pd.DataFrame([asdict(fp) for fp in historical])
    historical_path = os.path.join(output_dir, "fingerprints_historical.csv")
    historical_df.to_csv(historical_path, index=False)
    print(f"  Historical fingerprints saved to: {historical_path}")
    
    # Similar windows
    similar_data = []
    for result in similar:
        row = asdict(result.fingerprint)
        row['distance'] = result.distance
        row['rank'] = result.rank
        similar_data.append(row)
    
    similar_df = pd.DataFrame(similar_data)
    similar_path = os.path.join(output_dir, "fingerprints_similar.csv")
    similar_df.to_csv(similar_path, index=False)
    print(f"  Similar windows saved to: {similar_path}")


def export_to_json(current: Fingerprint, historical: List[Fingerprint], 
                   similar: List[SimilarityResult], output_dir: str):
    """
    Export fingerprints to JSON files.
    
    Args:
        current: Current fingerprint
        historical: List of all historical fingerprints
        similar: List of most similar windows
        output_dir: Output directory path
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Combined output
    output = {
        'generated_at': datetime.now().isoformat(),
        'current_fingerprint': asdict(current),
        'similar_windows': [
            {
                'rank': r.rank,
                'distance': r.distance,
                'fingerprint': asdict(r.fingerprint)
            }
            for r in similar
        ],
        'total_historical_windows': len(historical)
    }
    
    output_path = os.path.join(output_dir, "fingerprints.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"  Fingerprints saved to: {output_path}")
    
    # Full historical data (separate file due to size)
    historical_output = {
        'generated_at': datetime.now().isoformat(),
        'fingerprints': [asdict(fp) for fp in historical]
    }
    
    historical_path = os.path.join(output_dir, "fingerprints_historical_full.json")
    with open(historical_path, 'w') as f:
        json.dump(historical_output, f, indent=2)
    print(f"  Full historical data saved to: {historical_path}")


def print_fingerprint(fp: Fingerprint, title: str = "Fingerprint"):
    """Pretty-print a fingerprint"""
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    print(f"Window: {fp.window_start} to {fp.window_end} ({fp.window_size_days} days)")
    print(f"\n--- Volatility Features ---")
    print(f"  Avg Daily Range:        {fp.avg_daily_range:.6f}")
    print(f"  Avg ATR-14:             {fp.avg_atr_14:.6f}")
    print(f"  Avg BB Width:           {fp.avg_bb_width:.4f}")
    print(f"  Intraday Range % ATR:   {fp.intraday_range_pct_atr:.2f}%")
    print(f"\n--- Trend Features ---")
    print(f"  Avg Consecutive Up:     {fp.avg_consecutive_up_days:.2f} days")
    print(f"  Avg Consecutive Down:   {fp.avg_consecutive_down_days:.2f} days")
    print(f"  Max Consecutive Up:     {fp.max_consecutive_up_days} days")
    print(f"  Max Consecutive Down:   {fp.max_consecutive_down_days} days")
    print(f"  % Days Up:              {fp.pct_days_up:.1f}%")
    print(f"  % Days Down:            {fp.pct_days_down:.1f}%")
    print(f"\n--- Momentum Features ---")
    print(f"  % RSI Overbought:       {fp.pct_rsi_overbought:.1f}%")
    print(f"  % RSI Oversold:         {fp.pct_rsi_oversold:.1f}%")
    print(f"  % RSI Neutral:          {fp.pct_rsi_neutral:.1f}%")
    print(f"\n--- Volume Features ---")
    print(f"  Avg Daily Volume:       {fp.avg_daily_volume:,.2f}")
    print(f"\n--- Mean Reversion Features ---")
    print(f"  Avg Abs Dist SMA-50:    {fp.avg_abs_distance_sma_50:.4f}%")


def print_similar_windows(similar: List[SimilarityResult]):
    """Print top similar windows"""
    print(f"\n{'='*80}")
    print(f"TOP {len(similar)} SIMILAR HISTORICAL WINDOWS")
    print(f"{'='*80}")
    
    for result in similar:
        fp = result.fingerprint
        print(f"\n#{result.rank} | Distance: {result.distance:.4f} | Period: {fp.window_start} to {fp.window_end}")
        print(f"   Volatility: AvgRange={fp.avg_daily_range:.6f}, ATR={fp.avg_atr_14:.6f}, BBWidth={fp.avg_bb_width:.4f}")
        print(f"   Trend: %Up={fp.pct_days_up:.1f}%, %Down={fp.pct_days_down:.1f}%, MaxUp={fp.max_consecutive_up_days}d, MaxDown={fp.max_consecutive_down_days}d")
        print(f"   Momentum: RSI>70={fp.pct_rsi_overbought:.1f}%, RSI<30={fp.pct_rsi_oversold:.1f}%, RSI 40-60={fp.pct_rsi_neutral:.1f}%")
        print(f"   Volume: {fp.avg_daily_volume:,.0f} | MeanRev: {fp.avg_abs_distance_sma_50:.4f}%")


def main():
    """Main entry point for the fingerprint script"""
    parser = argparse.ArgumentParser(
        description='Calculate forex technical fingerprints and find similar historical periods'
    )
    parser.add_argument('--window-size', type=int, default=None,
                        help='Rolling window size in days (default: from config)')
    parser.add_argument('--window-step', type=int, default=None,
                        help='Rolling window step size in days (default: from config)')
    parser.add_argument('--top-n', type=int, default=None,
                        help='Number of top similar windows to return (default: from config)')
    parser.add_argument('--parallel-jobs', type=int, default=1,
                        help='Number of parallel jobs for historical calculation (default: 1)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for exports (default: generated/fingerprints)')
    parser.add_argument('--export-csv', action='store_true',
                        help='Export results to CSV files')
    parser.add_argument('--export-json', action='store_true',
                        help='Export results to JSON files')
    parser.add_argument('--data-file', type=str, default=None,
                        help='Path to daily OHLCV CSV file (default: eurusd_daily.csv)')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Suppress detailed output')
    
    args = parser.parse_args()
    
    # Load configuration
    config = get_fingerprint_config()
    
    # Override config with command line arguments
    window_size = args.window_size or config.get_window_size()
    window_step = args.window_step or config.get_window_step()
    top_n = args.top_n or config.get_top_n()
    params = config.get_parameters()
    rsi_thresholds = config.get_rsi_thresholds()
    
    # Set output directory
    output_dir = args.output_dir or os.path.join(dir_config.GENERATED_DIR, "fingerprints")
    
    # Set data file path
    data_file = args.data_file or os.path.join(dir_config.DATA_DIR, "eurusd_daily.csv")
    
    print(f"{'='*80}")
    print("FOREX TECHNICAL FINGERPRINT ANALYSIS")
    print(f"{'='*80}")
    print(f"Data file: {data_file}")
    print(f"Window size: {window_size} days")
    print(f"Window step: {window_step} days")
    print(f"Top N similar: {top_n}")
    print(f"Parallel jobs: {args.parallel_jobs}")
    print(f"{'='*80}")
    
    # Load data
    print("\n1. Loading data...")
    df = csv_utils.load_csv(data_file, filter_weekends_flag=True)
    print(f"   Loaded {len(df)} daily bars from {df.index.min()} to {df.index.max()}")
    
    # Ensure required columns exist
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Calculate indicators
    print("\n2. Calculating technical indicators...")
    df = calculate_indicators(df, params)
    
    # Drop rows with NaN values (indicator warmup period)
    initial_len = len(df)
    df = df.dropna()
    print(f"   Removed {initial_len - len(df)} rows (indicator warmup)")
    print(f"   Data range after warmup: {df.index.min()} to {df.index.max()}")
    
    # Check if we have enough data
    min_required = window_size + window_step + 1  # At least 2 windows
    if len(df) < min_required:
        raise ValueError(f"Not enough data: have {len(df)} rows, need at least {min_required}")
    
    # Calculate current fingerprint
    print("\n3. Calculating current fingerprint...")
    current_fp = calculate_current_fingerprint(df, window_size, rsi_thresholds)
    
    if not args.quiet:
        print_fingerprint(current_fp, "CURRENT FINGERPRINT")
    
    # Calculate historical fingerprints
    print("\n4. Calculating historical fingerprints...")
    historical_fps = calculate_historical_fingerprints(
        df, window_size, window_step, rsi_thresholds, args.parallel_jobs
    )
    print(f"   Calculated {len(historical_fps)} historical fingerprints")
    
    # Find similar windows
    print("\n5. Finding similar historical windows...")
    similar = find_similar_windows(current_fp, historical_fps, top_n)
    
    if not args.quiet:
        print_similar_windows(similar)
    
    # Export results
    if args.export_csv or args.export_json:
        print("\n6. Exporting results...")
        if args.export_csv:
            export_to_csv(current_fp, historical_fps, similar, output_dir)
        if args.export_json:
            export_to_json(current_fp, historical_fps, similar, output_dir)
    
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}")
    
    return {
        'current': current_fp,
        'historical': historical_fps,
        'similar': similar
    }


if __name__ == '__main__':
    main()
