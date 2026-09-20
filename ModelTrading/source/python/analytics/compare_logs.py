## Usage:
## Prepare backtest logs:
##      python .\backtest.py > C:\Users\mail\Daten\06_Projekte\github\forex_trading\ModelTrading\report\backtest\python_backtest.log 2>&1                                    
##      Run JForex backtest with logging enabled to generate java_backtest.log
## Compare logs:
##      python .\analytics\compare_logs.py C:\Users\mail\Daten\06_Projekte\github\forex_trading\ModelTrading\report\backtest\java_backtest.log C:\Users\mail\Daten\06_Projekte\github\forex_trading\ModelTrading\report\backtest\python_backtest.log

import re
import sys
import json
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict

# Timezone offset: JForex uses different timezone than CSV data
# Set this to the number of hours to subtract from Java timestamps when matching
# After testing: 0 appears correct - timestamps are actually aligned
JAVA_TIMEZONE_OFFSET_HOURS = 0  # Set to 0 if timestamps are aligned

# Define epsilon threshold for treating as zero difference (float32 precision limit)
# The 1e-5 threshold is reasonable - it's about 100x the true float32 machine epsilon, which accounts for accumulated errors across ~2-3 operations. 
# Making it larger (like 1e-4) would hide real algorithmic differences; 
# making it smaller would flag normal float32 variance.
epsilon = 1e-5  # 0.00001

def load_feature_names(config_path=None):
    """Load feature names from features.json. Returns dict with 'fast'/'slow' keys if scoped."""
    if config_path is None:
        possible_paths = [
            Path("ModelTrading/source/strategy/config/features.json"),
            Path("../../strategy/config/features.json"),
            Path("../../../config/features.json"),
            Path("C:/Program Files/JForex4/Strategies/files/production/jforex/features.json")
        ]
        for path in possible_paths:
            if path.exists():
                config_path = path
                break

    if config_path and Path(config_path).exists():
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                if 'scopes' in config:
                    result = {}
                    for scope, scope_obj in config['scopes'].items():
                        result[scope] = scope_obj.get('usedInModel_features', [])
                    return result
                if 'usedInModel_features' in config:
                    return config['usedInModel_features']
        except Exception as e:
            print(f"Warning: Could not load feature names from {config_path}: {e}")

    return None

def parse_java_log(file_path):
    """Parse Java log file into structured data"""
    entries = []
    
    # Try different encodings
    encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']
    lines = None
    
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                lines = f.readlines()
            break
        except UnicodeDecodeError:
            continue
    
    if lines is None:
        raise ValueError(f"Could not decode file {file_path} with any known encoding")
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Match prediction line - handle brackets and comma decimals
        pred_match = re.match(
            r'(?:\[[\d\-: ]+\]\s*)?(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}): Model predictions p_long_fast: ([\d.]+) \| p_short_fast: ([\d.]+) \| p_long_slow: ([\d.]+) \| p_short_slow: ([\d.]+) \| R: ([-\d.]+)',
            line
        )
        
        if pred_match:
            timestamp_str = pred_match.group(1)
            timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')

            # Convert comma decimals to periods
            entry = {
                'timestamp': timestamp,
                'p_long_fast': float(pred_match.group(2)),
                'p_short_fast': float(pred_match.group(3)),
                'p_long_slow': float(pred_match.group(4)),
                'p_short_slow': float(pred_match.group(5)),
                'r': float(pred_match.group(6)),
                'ohlc': None,
                'fast_features': None,
                'slow_features': None,
                'features': None
            }

            # Next line should be OHLC
            i += 1
            if i < len(lines):
                ohlc_line = lines[i].strip()
                ohlc_match = re.match(
                    r'(?:\[[\d\-: ]+\]\s*)?OHLC: O: ([\d.]+) \| H: ([\d.]+) \| L: ([\d.]+) \| C: ([\d.]+)',
                    ohlc_line
                )
                if ohlc_match:
                    entry['ohlc'] = {
                        'open': float(ohlc_match.group(1)),
                        'high': float(ohlc_match.group(2)),
                        'low': float(ohlc_match.group(3)),
                        'close': float(ohlc_match.group(4))
                    }

            # Next lines should be Fast features and Slow features (or legacy Features)
            i += 1
            if i < len(lines):
                features_line = lines[i].strip()
                fast_match = re.match(r'(?:\[[\d\-: ]+\]\s*)?Fast features: \[(.*)\]', features_line)
                legacy_match = re.match(r'(?:\[[\d\-: ]+\]\s*)?Features: \[(.*)\]', features_line)
                if fast_match:
                    features_str = fast_match.group(1)
                    entry['fast_features'] = [float(x.strip()) for x in features_str.split(', ')]
                    # Next line should be Slow features
                    i += 1
                    if i < len(lines):
                        slow_line = lines[i].strip()
                        slow_match = re.match(r'(?:\[[\d\-: ]+\]\s*)?Slow features: \[(.*)\]', slow_line)
                        if slow_match:
                            features_str = slow_match.group(1)
                            entry['slow_features'] = [float(x.strip()) for x in features_str.split(', ')]
                elif legacy_match:
                    features_str = legacy_match.group(1)
                    entry['features'] = [float(x.strip()) for x in features_str.split(', ')]

            entries.append(entry)
        
        i += 1
    
    return entries

def parse_python_log(file_path):
    """Parse Python log file into structured data"""
    entries = []
    
    # Try different encodings - UTF-16 first for Windows PowerShell redirects
    encodings = ['utf-16', 'utf-16-le', 'utf-8', 'utf-8-sig', 'latin-1', 'cp1252']
    lines = None
    
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                lines = f.readlines()
            break
        except (UnicodeDecodeError, UnicodeError):
            continue
    
    if lines is None:
        raise ValueError(f"Could not decode file {file_path} with any known encoding")
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Match prediction line - more flexible pattern
        pred_match = re.match(
            r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}): Model predictions p_long_fast: ([\d.]+) \| p_short_fast: ([\d.]+) \| p_long_slow: ([\d.]+) \| p_short_slow: ([\d.]+) \| R: (-?[\d.]+)',
            line
        )
        
        if pred_match:
            timestamp_str = pred_match.group(1)
            timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')

            entry = {
                'timestamp': timestamp,
                'p_long_fast': float(pred_match.group(2)),
                'p_short_fast': float(pred_match.group(3)),
                'p_long_slow': float(pred_match.group(4)),
                'p_short_slow': float(pred_match.group(5)),
                'r': float(pred_match.group(6)),
                'ohlc': None,
                'fast_features': None,
                'slow_features': None,
                'features': None
            }

            # Next line should be OHLC (may have leading spaces)
            i += 1
            if i < len(lines):
                ohlc_line = lines[i].strip()
                ohlc_match = re.match(
                    r'OHLC: O: ([\d.]+) \| H: ([\d.]+) \| L: ([\d.]+) \| C: ([\d.]+)',
                    ohlc_line
                )
                if ohlc_match:
                    entry['ohlc'] = {
                        'open': float(ohlc_match.group(1)),
                        'high': float(ohlc_match.group(2)),
                        'low': float(ohlc_match.group(3)),
                        'close': float(ohlc_match.group(4))
                    }

            # Next lines: Fast features and Slow features (or legacy Features)
            i += 1
            if i < len(lines):
                features_line = lines[i].strip()
                fast_match = re.match(r'Fast features: \[(.*)\]', features_line)
                legacy_match = re.match(r'Features: \[(.*)\]', features_line)
                if fast_match:
                    entry['fast_features'] = [float(x.strip()) for x in fast_match.group(1).split(',')]
                    # Next line should be Slow features
                    i += 1
                    if i < len(lines):
                        slow_line = lines[i].strip()
                        slow_match = re.match(r'Slow features: \[(.*)\]', slow_line)
                        if slow_match:
                            entry['slow_features'] = [float(x.strip()) for x in slow_match.group(1).split(',')]
                elif legacy_match:
                    entry['features'] = [float(x.strip()) for x in legacy_match.group(1).split(',')]

            entries.append(entry)
        
        i += 1
    
    return entries

def compare_entries(java_entry, python_entry, tolerance, feature_stats, feature_names):
    """Compare two entries and return differences"""
    differences = []
    
    # Compare predictions
    for key in ['p_long_fast', 'p_short_fast', 'p_long_slow', 'p_short_slow', 'r']:
        java_val = java_entry[key]
        python_val = python_entry[key]
        diff = abs(java_val - python_val)
        
        # Treat differences below epsilon as zero
        if diff < epsilon:
            diff = 0.0
        
        if diff > tolerance:
            differences.append(f"  {key}: Java={java_val:.6f}, Python={python_val:.6f}, diff={diff:.9f}")
    
    # Compare OHLC
    if java_entry['ohlc'] and python_entry['ohlc']:
        for key in ['open', 'high', 'low', 'close']:
            java_val = java_entry['ohlc'][key]
            python_val = python_entry['ohlc'][key]
            diff = abs(java_val - python_val)
            
            # Treat differences below epsilon as zero
            if diff < epsilon:
                diff = 0.0
            
            if diff > tolerance:
                differences.append(f"  OHLC.{key}: Java={java_val:.5f}, Python={python_val:.5f}, diff={diff:.9f}")
    
    # Compare fast + slow features (new split format) or legacy single features
    def compare_feature_list(java_feats, python_feats, prefix, names_list):
        if java_feats is None or python_feats is None:
            return
        if len(java_feats) != len(python_feats):
            differences.append(f"  {prefix} count mismatch: Java={len(java_feats)}, Python={len(python_feats)}")
            return
        for idx, (java_val, python_val) in enumerate(zip(java_feats, python_feats)):
            diff = abs(java_val - python_val)
            if diff < epsilon:
                diff = 0.0
            stat_key = f"{prefix}[{idx}]"
            feature_stats[stat_key]['count'] += 1
            feature_stats[stat_key]['total_diff'] += diff
            feature_stats[stat_key]['max_diff'] = max(feature_stats[stat_key]['max_diff'], diff)
            if diff > tolerance:
                feature_stats[stat_key]['mismatch_count'] += 1
            if diff > tolerance:
                feat_name = names_list[idx] if names_list and idx < len(names_list) else f"{prefix}[{idx}]"
                differences.append(f"  {feat_name}: Java={java_val:.9f}, Python={python_val:.9f}, diff={diff:.9f}")

    if java_entry.get('fast_features') or python_entry.get('fast_features'):
        compare_feature_list(java_entry.get('fast_features'), python_entry.get('fast_features'),
                             'fast', feature_names.get('fast') if isinstance(feature_names, dict) else feature_names)
        compare_feature_list(java_entry.get('slow_features'), python_entry.get('slow_features'),
                             'slow', feature_names.get('slow') if isinstance(feature_names, dict) else None)
    elif java_entry.get('features') and python_entry.get('features'):
        names_list = feature_names if not isinstance(feature_names, dict) else None
        compare_feature_list(java_entry['features'], python_entry['features'], 'feat', names_list)

    return differences

def main():
    if len(sys.argv) < 3:
        print("Usage: python compare_backtest_logs.py <java_log_file> <python_log_file> [features_json_path]")
        print("\nExample:")
        print("  python compare_backtest_logs.py java_backtest.log python_backtest.log")
        print("  python compare_backtest_logs.py java_backtest.log python_backtest.log path/to/features.json")
        sys.exit(1)
    
    java_log_path = Path(sys.argv[1])
    python_log_path = Path(sys.argv[2])
    features_json_path = Path(sys.argv[3]) if len(sys.argv) > 3 else None
    
    if not java_log_path.exists():
        print(f"Error: Java log file not found: {java_log_path}")
        sys.exit(1)
    
    if not python_log_path.exists():
        print(f"Error: Python log file not found: {python_log_path}")
        sys.exit(1)
    
    # Load feature names
    print("Loading feature names...")
    feature_names = load_feature_names(features_json_path)
    if feature_names:
        print(f"[OK] Loaded {len(feature_names)} feature names from configuration")
    else:
        print("[!] Could not load feature names, will use indices")
    
    print("Parsing Java log...")
    java_entries = parse_java_log(java_log_path)
    print(f"Found {len(java_entries)} Java entries")
    
    print("Parsing Python log...")
    python_entries = parse_python_log(python_log_path)
    print(f"Found {len(python_entries)} Python entries")
    
    # Create timestamp index for quick lookup
    python_by_timestamp = {e['timestamp']: e for e in python_entries}
    
    # Feature statistics tracking
    feature_stats = defaultdict(lambda: {
        'count': 0,
        'mismatch_count': 0,
        'total_diff': 0.0,
        'max_diff': 0.0
    })
    
    tolerance = 1e-6
    
    print("\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*80)
    if JAVA_TIMEZONE_OFFSET_HOURS != 0:
        print(f"⏰ Applying timezone offset: Java timestamps - {JAVA_TIMEZONE_OFFSET_HOURS} hours")

    mismatches = 0
    missing_in_python = 0
    first_mismatch_timestamp = None
    
    for java_entry in java_entries:
        java_timestamp = java_entry['timestamp']
        # Apply timezone offset: convert Java timestamp to Python timestamp
        # Java timestamps appear to be +3 hours ahead, so subtract 3 hours
        python_timestamp = java_timestamp - timedelta(hours=JAVA_TIMEZONE_OFFSET_HOURS)

        if python_timestamp not in python_by_timestamp:
            missing_in_python += 1
            if missing_in_python <= 5:
                print(f"\n[!] MISSING in Python: Java {java_timestamp} -> Python {python_timestamp}")
            continue

        python_entry = python_by_timestamp[python_timestamp]
        differences = compare_entries(java_entry, python_entry, tolerance, feature_stats, feature_names)
        
        if differences:
            mismatches += 1
            if first_mismatch_timestamp is None:
                first_mismatch_timestamp = java_timestamp

            if mismatches <= 10:  # Show first 10 mismatches
                print(f"\n[MISMATCH] at Java {java_timestamp} (Python {python_timestamp}):")
                for diff in differences:
                    print(diff)
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total Java entries:      {len(java_entries)}")
    print(f"Total Python entries:    {len(python_entries)}")
    print(f"Missing in Python:       {missing_in_python}")
    print(f"Entries with mismatches: {mismatches}")
    
    if first_mismatch_timestamp:
        print(f"\n[!] First mismatch at:    {first_mismatch_timestamp}")
    
    if mismatches == 0 and missing_in_python == 0:
        print("\n[OK] All entries match perfectly!")
    else:
        print(f"\n[!] Found {mismatches} entries with differences")
        if mismatches > 10:
            print(f"   (showing first 10, {mismatches - 10} more not displayed)")
    
    # Feature deviation summary
    if feature_stats:
        print("\n" + "="*80)
        print("FEATURE DEVIATION SUMMARY")
        print("="*80)
        print(f"{'Feature':<40} {'Mismatches':<12} {'Mismatch %':<12} {'Avg Diff':<15} {'Max Diff':<15}")
        print("-" * 110)

        sorted_features = sorted(feature_stats.items(),
                                 key=lambda x: (x[1]['mismatch_count'], x[1]['max_diff']),
                                 reverse=True)

        for feature_key, stats in sorted_features[:99]:
            if stats['count'] > 0 and stats['mismatch_count'] > 0:
                mismatch_pct = (stats['mismatch_count'] / stats['count']) * 100
                avg_diff = stats['total_diff'] / stats['count']
                # feature_key is a string like "fast[0]" or "slow[2]"
                print(f"{str(feature_key):<40} {stats['mismatch_count']:<12} {mismatch_pct:<12.2f} {avg_diff:<15.9f} {stats['max_diff']:<15.9f}")

        if len(sorted_features) > 99:
            print(f"\n... and {len(sorted_features) - 99} more features")
    
    features_within_epsilon = sum(1 for stats in feature_stats.values() 
                                   if stats['max_diff'] < epsilon and stats['count'] > 0)
    total_features = len([s for s in feature_stats.values() if s['count'] > 0])
    
    print(f"\n[OK] {features_within_epsilon}/{total_features} features within float32 precision (< 1e-5)")
    
    print("="*80)

if __name__ == "__main__":
    main()