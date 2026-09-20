import os
from datetime import datetime
import pandas as pd
import ModelTrading.config.directories as dir_config

def plot_label_distribution(y_cls_long, y_cls_short, df, pip_value=0.01, max_plots=9999999, long_fast=None, short_fast=None, last_weeks=None):
    """
    Existing behaviour: plot slow labels (long/short) on price.

    Extended: if long_fast/short_fast are provided, overlay them on the SAME figure
    so the same saved artefacts are produced (same filenames/locations), just with
    additional markers.
    """
    import matplotlib.pyplot as plt

    # Convert label indices to datetime for comparison (if they're strings)
    if not isinstance(y_cls_long.index, pd.DatetimeIndex):
        y_cls_long = y_cls_long.copy()
        y_cls_long.index = pd.to_datetime(y_cls_long.index, format='%d.%m.%Y %H:%M')

    if not isinstance(y_cls_short.index, pd.DatetimeIndex):
        y_cls_short = y_cls_short.copy()
        y_cls_short.index = pd.to_datetime(y_cls_short.index, format='%d.%m.%Y %H:%M')

    # NEW: also normalize fast label indices (if provided)
    if long_fast is not None and not isinstance(long_fast.index, pd.DatetimeIndex):
        long_fast = long_fast.copy()
        long_fast.index = pd.to_datetime(long_fast.index, format='%d.%m.%Y %H:%M')

    if short_fast is not None and not isinstance(short_fast.index, pd.DatetimeIndex):
        short_fast = short_fast.copy()
        short_fast.index = pd.to_datetime(short_fast.index, format='%d.%m.%Y %H:%M')

    # Optional: restrict to the most recent N weeks
    if last_weeks is not None and last_weeks > 0:
        df_index_dt = df.index
        if len(df_index_dt) > 0 and isinstance(df_index_dt[0], str):
            df_index_dt = pd.to_datetime(df_index_dt, format='%d.%m.%Y %H:%M')
        else:
            df_index_dt = pd.DatetimeIndex(df_index_dt)
        if len(df_index_dt) > 0:
            cutoff = df_index_dt.max() - pd.Timedelta(weeks=last_weeks)
            mask = df_index_dt > cutoff
            df = df[mask]
            print(f"plot_label_distribution: restricted to last {last_weeks} weeks "
                  f"(cutoff={cutoff}, {len(df)} bars).")

    # Gruppiere nach Monat
    monthly_groups = {}
    for idx in df.index:
        if isinstance(idx, str):
            idx_dt = datetime.strptime(idx, '%d.%m.%Y %H:%M')
            month_key = idx_dt.strftime('%Y-%m')
        else:
            month_key = idx.strftime('%Y-%m')
        if month_key not in monthly_groups:
            monthly_groups[month_key] = []
        monthly_groups[month_key].append(idx)
    
    # Plotte jede Woche in separater Datei
    plot_count = 0
    for month_key, indices in sorted(monthly_groups.items()):
        if plot_count >= max_plots:
            break
            
        if not indices:
            continue
        
        # Filter Daten für diesen Monat
        year, month = map(int, month_key.split('-'))
        month_start = datetime(year, month, 1)
        if month == 12:
            month_end = datetime(year + 1, 1, 1)
        else:
            month_end = datetime(year, month + 1, 1)
        
        # Filtere df nach diesem Zeitraum
        df_index = df.index
        if len(df_index) > 0 and isinstance(df_index[0], str):
            df_index = [datetime.strptime(idx, '%d.%m.%Y %H:%M') for idx in df_index]
        mask = [(idx >= month_start and idx < month_end) for idx in df_index]
        month_data = df[mask].copy()
        
        if len(month_data) == 0:
            continue
        
        # Gruppiere nach Woche
        month_data_indexed = month_data.copy()
        if isinstance(month_data_indexed.index[0], str):
            month_data_indexed.index = pd.to_datetime(month_data_indexed.index, format='%d.%m.%Y %H:%M')

        # Gruppiere nach ISO Kalenderwoche
        for week_num, week_group in month_data_indexed.groupby(month_data_indexed.index.isocalendar().week):
            if plot_count >= max_plots:
                break
                
            window = week_group.copy()
            
            if len(window) == 0:
                continue

            # Remove rows with zero volume or identical OHLC (no trading activity)
            if 'volume' in window.columns:
                window = window[window['volume'] > 0]
            else:
                window = window[~((window['open'] == window['high']) & 
                                (window['high'] == window['low']) & 
                                (window['low'] == window['close']))]

            if len(window) == 0:
                continue
            
            # Get week start/end dates
            week_start = window.index.min().strftime('%Y-%m-%d')
            week_end = window.index.max().strftime('%Y-%m-%d')
            
            # Reset index to use sequential integers
            window = window.reset_index()
            date_column = window.columns[0]
            
            # Calculate figure width and bar width with spacing
            bar_width = 0.8  # Leave 0.2 units gap between bars
            min_width_per_bar = 3  # pixels per bar to ensure spacing
            fig_width = max(20, len(window) * min_width_per_bar / 100)  # Convert to inches (100 dpi)
            fig_width = min(fig_width, 100)  # Cap at 100 inches

            fig, ax1 = plt.subplots(figsize=(fig_width, 8))
            
            # Plot candlestick bars using sequential index with spacing
            x_positions = list(range(len(window)))

            for i in x_positions:
                color = 'green' if window['close'].iloc[i] >= window['open'].iloc[i] else 'red'
                # Draw wick (high-low line)
                ax1.plot([i, i], [window['low'].iloc[i], window['high'].iloc[i]], color=color, linewidth=1, zorder=1)
                # Draw body as rectangle with width < 1 to create spacing
                body_height = abs(window['close'].iloc[i] - window['open'].iloc[i])
                body_bottom = min(window['open'].iloc[i], window['close'].iloc[i])
                ax1.bar(i, body_height, bottom=body_bottom, width=bar_width, color=color, edgecolor=color, zorder=2)

            # Set y-axis limits for price chart with some padding
            price_min = window['low'].min()
            price_max = window['high'].max()
            price_range = price_max - price_min
            padding_top = price_range * 0.05  # 5% padding
            padding_bottom = price_range * 0.15  # 15% padding
            ax1.set_ylim(price_min - padding_bottom, price_max + padding_top)

            # Create secondary y-axis for labels
            ax2 = ax1.twinx()
            ax2.set_ylim(0, 20)
            ax2.set_ylabel('Labels (1=Signal)', color='black')
            ax2.tick_params(axis='y', labelcolor='black')

            # Collect label positions
            long_positions = []
            short_positions = []
            fast_long_positions = []
            fast_short_positions = []
            label_count = 0

            for i in x_positions:
                timestamp = window[date_column].iloc[i]

                # Convert timestamp to comparable format
                if isinstance(timestamp, str):
                    timestamp = pd.to_datetime(timestamp, format='%d.%m.%Y %H:%M')

                # Slow labels
                if timestamp in y_cls_long.index and y_cls_long.loc[timestamp] == 1:
                    long_positions.append(i)
                    label_count += 1

                if timestamp in y_cls_short.index and y_cls_short.loc[timestamp] == 1:
                    short_positions.append(i)
                    label_count += 1

                # Fast labels (optional)
                if long_fast is not None and timestamp in long_fast.index and long_fast.loc[timestamp] == 1:
                    fast_long_positions.append(i)
                    label_count += 1

                if short_fast is not None and timestamp in short_fast.index and short_fast.loc[timestamp] == 1:
                    fast_short_positions.append(i)
                    label_count += 1

            # Plot slow labels on secondary axis
            if long_positions:
                ax2.scatter(
                    long_positions, [1] * len(long_positions),
                    marker='o', color='lime', s=100, zorder=5,
                    edgecolor='darkgreen', linewidth=2, label='Long Slow Signal'
                )

            if fast_long_positions:
                ax2.scatter(
                    fast_long_positions, [2] * len(fast_long_positions),
                    marker='x', color='lime', s=80, zorder=6,
                    linewidth=2, label='Long Fast Signal'
                )

            if short_positions:
                ax2.scatter(
                    short_positions, [3] * len(short_positions),
                    marker='o', color='red', s=100, zorder=5,
                    edgecolor='darkred', linewidth=2, label='Short SlowSignal'
                )

            if fast_short_positions:
                ax2.scatter(
                    fast_short_positions, [4] * len(fast_short_positions),
                    marker='x', color='red', s=80, zorder=6,
                    linewidth=2, label='Short Fast Signal'
                )

            if long_positions or short_positions or fast_long_positions or fast_short_positions:
                ax2.legend(loc='upper right')

            # Set x-axis labels at regular intervals (showing actual dates)
            num_ticks = min(20, len(window))
            if num_ticks > 0:
                tick_positions = [int(i * (len(window) - 1) / (num_ticks - 1)) for i in range(num_ticks)]
                unique_positions = []
                tick_labels = []
                seen_dates = set()

                for pos in tick_positions:
                    date_val = window[date_column].iloc[pos]
                    date_str = date_val.strftime('%Y-%m-%d') if hasattr(date_val, 'strftime') else str(date_val).split()[0]

                    if date_str in seen_dates:
                        continue

                    seen_dates.add(date_str)
                    unique_positions.append(pos)
                    tick_labels.append(date_str)

                if unique_positions:
                    ax1.set_xticks(unique_positions)
                    ax1.set_xticklabels(tick_labels, rotation=45, ha='right')
            
            ax1.set_title(f"Week {week_num}: {week_start} to {week_end}")
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Price', color='black')
            plt.tight_layout()
            plt.savefig(f"{dir_config.LABEL_REVIEW_DIR}/price_data_{week_start}_week_{week_num}.png", dpi=100)
            plt.close()
            plot_count += 1