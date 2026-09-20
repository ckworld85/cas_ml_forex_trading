def remove_duplicates(df, name="DataFrame", removal_strategy='last'):
    """Remove duplicate index entries, keeping last occurrence, then ensure sorted"""
    if df.index.duplicated().any():
        n_duplicates = df.index.duplicated().sum()
        print(f"WARNING - remove_duplicates(): Found {n_duplicates} duplicate timestamps in {name}")
        df = df[~df.index.duplicated(keep=removal_strategy)]
        print(f"  Removed duplicates, keeping last occurrence. New length: {len(df)}")
        df = df.sort_index()
        print(f"  Re-sorted {name} after duplicate removal")
    return df

def datacheck_chronological_order(df=None, name="DataFrame"):
    if df is None:
        print("WARNING - datacheck_chronological_order(): No dataframes provided for chronological order check.")
        return

    if not df.index.is_monotonic_increasing:
        print(f"WARNING: {name} is NOT chronologically sorted!")
        all_sorted = False
        # Force sort
        df = df.sort_index()
        print(f"-> Fixed: {name} has been sorted")
    else:
        print(f"OK: {name}: is already chronologically sorted")

    return df