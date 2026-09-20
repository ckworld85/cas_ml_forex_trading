import os
import pandas as pd
import numpy as np
import ModelTrading.config.directories as dir_config
import ModelTrading.source.python.utils.datahandling as datahandling

def load_csv(path, start_date=None, end_date=None, filter_weekends_flag=True,
             keep_mid=False):
    """Load a provider OHLC CSV.

    Args:
        keep_mid: retain the 'mid' column instead of dropping it. Default False so the
            feature pipeline keeps seeing exactly the columns it always saw. The
            transaction-cost model (utils/costs.py) sets it True: the spread is
            2*(ask_close - mid) and there is no other source for it.
    """
    df = pd.read_csv(path, delimiter=",")
    
    # Try to detect and parse the date format
    sample_date = str(df['time'].iloc[0])
    
    if '.' in sample_date:
        # European format: dd.mm.yyyy HH:MM
        df['time'] = pd.to_datetime(df['time'], format='%d.%m.%Y %H:%M', dayfirst=True)
    else:
        # ISO format: yyyy-mm-dd HH:MM:SS or similar. utc=True is required
        # because a file can legitimately MIX tz-aware ('...Z') and naive
        # rows (the non-EURUSD pair exports carry both generations); naive
        # rows are treated as UTC, which is what the pipeline always assumed.
        df['time'] = pd.to_datetime(df['time'], format='ISO8601', utc=True)

    df.set_index("time", inplace=True)
    # Normalize to tz-naive UTC — strip timezone info if present (e.g. from ISO 8601 'Z' suffix)
    if df.index.tz is not None:
        df.index = df.index.tz_convert('UTC').tz_localize(None)
    df.sort_index(inplace=True)

    # Verify the sort actually worked
    if not df.index.is_monotonic_increasing:
        print(f"  ERROR: {os.path.basename(path)} is NOT sorted after sort_index()!")
        print(f"  First 5 timestamps: {df.index[:5].tolist()}")
        print(f"  Last 5 timestamps: {df.index[-5:].tolist()}")

    # Data-integrity tripwire: duplicate timestamps whose values DISAGREE are
    # corruption, not harmless re-exports — measured 2026-09-05: 16 bars of
    # OTHER instruments (JPY/AUD/NZD prices) sat inside eurusd_m15.csv under
    # duplicated timestamps, mislabeled at the DB source. remove_duplicates
    # keeps an arbitrary one of the pair (keep='last' after an unstable sort),
    # so a silent pass here would let a foreign bar replace a real one.
    dup_mask = df.index.duplicated(keep=False)
    if dup_mask.any():
        n_conflicts = int((df.loc[dup_mask].groupby(level=0)['close'].nunique() > 1).sum())
        if n_conflicts > 0:
            conflicting = df.loc[dup_mask].groupby(level=0)['close'].nunique()
            examples = conflicting[conflicting > 1].index[:5].tolist()
            print(f"  WARNING - CORRUPT DATA in {os.path.basename(path)}: "
                  f"{n_conflicts} duplicate timestamps carry CONFLICTING values "
                  f"(e.g. {examples}). These are likely bars of another "
                  f"instrument mislabeled at the source — the kept row is "
                  f"arbitrary. Fix the export/database, do not trust this file.")

    # Remove duplicate timestamps if any
    df = datahandling.remove_duplicates(df, os.path.basename(path))

    # Keep OHLC as float64 for accurate feature computation (ATR, BB, etc.)
    # features.py converts computed features to float32 at the end.
    # Premature float32 cast here causes ~0.04% ATR error vs Java (float64).

    # Filter weekends
    if filter_weekends_flag:
        initial_count = len(df)
        df = df[(df.index.weekday < 4) | ((df.index.weekday == 4) & (df.index.hour < 22))].copy()  # Keep Monday (0) to Friday (4) and Friday only before 22:00
        removed_count = initial_count - len(df)
        if removed_count > 0:
            print(f"  Filtered out {removed_count} weekend bars ({removed_count/initial_count*100:.2f}%)")
    
    # Apply date filters
    if start_date is not None:
        df = df[df.index >= start_date]
        print(f"  Data loaded from {os.path.basename(path)}, filtered to start at {start_date} (actual start: {df.index.min()})")
    if end_date is not None:
        df = df[df.index <= end_date]
        print(f"  Data loaded from {os.path.basename(path)}, filtered to end at {end_date} (actual end: {df.index.max()})")

    ## remove column mid from df (unless the caller needs it for the spread)
    if 'mid' in df.columns and not keep_mid:
        df.drop(columns=['mid'], inplace=True)

    return df

def export_training_data_to_csv(X_train,
                                ohlc_train,
                                y_train_cls_long,
                                y_train_cls_short,
                                y_train_target_long,
                                y_train_target_short,
                                scaler, output_dir,
                                y_train_cls_long_raw=None,
                                y_train_cls_short_raw=None):
    """
    Export training and test data EXACTLY as used by the models.
    This includes OHLC, features, and all targets.
    Optionally includes raw (pre-denoising) labels for review.
    """
    
    # Export TRAIN data
    train_export = pd.DataFrame(index=X_train.index)
    
    # Add OHLC first (without m15_ prefix for cleaner columns)
    train_export['open'] = ohlc_train.iloc[:, 0]
    train_export['high'] = ohlc_train.iloc[:, 1]
    train_export['low'] = ohlc_train.iloc[:, 2]
    train_export['close'] = ohlc_train.iloc[:, 3]
    
    # Add targets
    train_export['y_cls_long'] = y_train_cls_long
    train_export['y_cls_short'] = y_train_cls_short
    train_export['y_hit_pips_long'] = y_train_target_long
    train_export['y_hit_pips_short'] = y_train_target_short

    # Add raw (pre-denoising) labels if available for comparison
    if y_train_cls_long_raw is not None:
        train_export['y_cls_long_raw'] = y_train_cls_long_raw
    if y_train_cls_short_raw is not None:
        train_export['y_cls_short_raw'] = y_train_cls_short_raw
        
    # Sort chronologically
    train_export = train_export.sort_index()
    
    # Export combined dataset
    csv_path = os.path.join(output_dir, "training_data_full.csv")
    train_export.to_csv(csv_path, index=True, date_format='%Y-%m-%d %H:%M:%S')
    print(f"\nOK: Full training data exported to: {csv_path}")
    print(f"  Train rows: {len(train_export)}")
    print(f"  Chronologically sorted: {train_export.index.is_monotonic_increasing}")
    
    # Also export just train separately for convenience
    train_csv = os.path.join(output_dir, "training_data_train_only.csv")
    
    train_export.to_csv(train_csv, index=True, date_format='%Y-%m-%d %H:%M:%S')
    
    print(f"  Train-only: {train_csv}")
    
    # Export a summary
    summary = {
        'train_date_range': f"{train_export.index.min()} to {train_export.index.max()}",
        'train_samples': len(train_export),
        'train_long_labels': y_train_cls_long.sum(),
        'train_short_labels': y_train_cls_short.sum(),
        'feature_count': len(X_train.columns),
        'train_sorted': train_export.index.is_monotonic_increasing,
    }
    
    summary_path = os.path.join(output_dir, "training_data_summary.txt")
    with open(summary_path, 'w') as f:
        for key, value in summary.items():
            f.write(f"{key}: {value}\n")
    
    print(f"  Summary: {summary_path}\n")

def filter_weekends(df):
    """
    Filter out bars that occur on weekends (Saturday/Sunday).
    This mimics JForex Filter.WEEKENDS behavior.
    
    Args:
        df: DataFrame with DatetimeIndex
        
    Returns:
        DataFrame with weekend bars removed
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        print("Warning: filter_weekends requires DatetimeIndex")
        return df
    
    initial_count = len(df)
    
    # Filter: Keep only Monday (0) through Friday (4)
    # weekday(): Monday=0, Sunday=6
    df_filtered = df[df.index.weekday < 5].copy()
    
    removed_count = initial_count - len(df_filtered)
    
    if removed_count > 0:
        print(f"  Filtered out {removed_count} weekend bars ({removed_count/initial_count*100:.2f}%)")
    
    return df_filtered