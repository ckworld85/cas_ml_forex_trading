"""
Data Update Script

Fetches data from MySQL database views and exports to CSV files.
Configuration-driven approach allows easy addition of new data sources.
"""

import os
import sys
import yaml
import mysql.connector
import pandas as pd
import requests
from datetime import datetime

# Add project root to Python path to enable ModelTrading package imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import ModelTrading.config.directories as dir_config

# Directory paths
BASE_DIR = dir_config.BASE_DIR
DATA_DIR = dir_config.DATA_DIR
CONFIG_DIR = dir_config.CONFIG_DIR
DB_CONFIG_PATH = os.path.join(CONFIG_DIR, "db_config.yaml")

# Instruments available in the export views. The `instrument` column of every view
# listed in DATA_EXPORTS is the selection criterion, and the output file name is
# derived from it: the slash is removed and the rest lower-cased
# ("EUR/USD" -> "eurusd" -> eurusd_m15.csv).
INSTRUMENTS = [
    "AUD/JPY",
    "AUD/USD",
    "EUR/GBP",
    "EUR/JPY",
    "EUR/USD",
    "GBP/USD",
    "NZD/USD",
    "USD/CAD",
    "USD/CHF",
    "USD/JPY",
]

# Name of the selection column present in every export view
INSTRUMENT_COLUMN = "instrument"

# Columns selected from every export view. Listed explicitly instead of "SELECT *"
# so a new column in a view cannot silently change the CSV layout the feature
# pipeline reads (utils/csv.load_csv).
EXPORT_COLUMNS = ["time", "open", "high", "low", "close", "volume", "mid"]

# Data export configuration
# Each entry maps a MySQL view to an output CSV file. `output_file` is a template;
# `{instrument}` is replaced by the slug of the instrument currently being fetched.
DATA_EXPORTS = [
    {
        "name": "M15 Bars",
        "view": "bars_15min_export",
        "output_file": "{instrument}_m15.csv",
        "enabled": True,
    },
    {
        "name": "Daily Bars",
        "view": "bars_daily_export",
        "output_file": "{instrument}_daily.csv",
        "enabled": True,
    },
    {
        "name": "4 Hours Bars",
        "view": "bars_4hours_export",
        "output_file": "{instrument}_4hours.csv",
        "enabled": True,
    },
    # Add more exports here as needed:
    # {
    #     "name": "Hourly Bars",
    #     "view": "bars_hourly_export",
    #     "output_file": "{instrument}_h1.csv",
    #     "enabled": True,
    # },
]


def instrument_slug(instrument):
    """Return the file-name form of an instrument: 'EUR/USD' -> 'eurusd'."""
    return instrument.replace("/", "").strip().lower()


def resolve_output_file(template, instrument):
    """Fill the '{instrument}' placeholder of an output-file template.

    Templates without the placeholder are returned unchanged, so a fixed file
    name still works.
    """
    return template.format(instrument=instrument_slug(instrument))


def normalize_instrument(value):
    """Map a user-supplied instrument onto its INSTRUMENTS entry.

    Accepts any spelling that has the same slug ('eurusd', 'EUR/USD', 'eur/usd').
    Returns None if the instrument is not configured.
    """
    slug = instrument_slug(value)
    for instrument in INSTRUMENTS:
        if instrument_slug(instrument) == slug:
            return instrument
    return None


def get_current_ip():
    """Get the current public IP address."""
    try:
        response = requests.get('https://api.ipify.org?format=json', timeout=10)
        response.raise_for_status()
        return response.json()['ip']
    except Exception as e:
        print(f"Error getting current IP: {e}")
        return None


def whitelist_ip_cpanel(ip_address):
    """
    Whitelist an IP address for MySQL access via cPanel API.
    
    Requires environment variables:
    - CPANEL_HOST: Your cPanel hostname (e.g., yourdomain.com)
    - CPANEL_USER: Your cPanel username
    - CPANEL_API_TOKEN: Your cPanel API token
    """
    cpanel_host = os.environ.get('CPANEL_HOST')
    cpanel_user = os.environ.get('CPANEL_USER')
    cpanel_token = os.environ.get('CPANEL_API_TOKEN')
    
    if not all([cpanel_host, cpanel_user, cpanel_token]):
        print("Warning: cPanel credentials not found. Skipping IP whitelisting.")
        return False
    
    try:
        # cPanel API endpoint for adding remote MySQL host
        url = f"https://{cpanel_host}:2083/execute/Mysql/add_host"
        
        headers = {
            'Authorization': f'cpanel {cpanel_user}:{cpanel_token}',
        }
        
        params = {
            'host': ip_address,
        }
        
        response = requests.get(url, headers=headers, params=params, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        
        if result.get('status') == 1:
            print(f"Successfully whitelisted IP: {ip_address}")
            return True
        else:
            print(f"Failed to whitelist IP: {result.get('errors', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"Error whitelisting IP via cPanel: {e}")
        return False


def remove_ip_cpanel(ip_address):
    """
    Remove an IP address from MySQL whitelist via cPanel API.
    """
    cpanel_host = os.environ.get('CPANEL_HOST')
    cpanel_user = os.environ.get('CPANEL_USER')
    cpanel_token = os.environ.get('CPANEL_API_TOKEN')
    
    if not all([cpanel_host, cpanel_user, cpanel_token]):
        return False
    
    try:
        url = f"https://{cpanel_host}:2083/execute/Mysql/delete_host"
        
        headers = {
            'Authorization': f'cpanel {cpanel_user}:{cpanel_token}',
        }
        
        params = {
            'host': ip_address,
        }
        
        response = requests.get(url, headers=headers, params=params, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        
        if result.get('status') == 1:
            print(f"Successfully removed IP from whitelist: {ip_address}")
            return True
        else:
            print(f"Failed to remove IP: {result.get('errors', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"Error removing IP via cPanel: {e}")
        return False


def load_db_config():
    """
    Load database configuration from environment variables or YAML file.
    
    Priority:
    1. Environment variables (GitHub Actions secrets)
    2. YAML configuration file (local development)
    """
    # Check if environment variables are set (GitHub Actions case)
    env_vars = {
        'host': os.environ.get('DB_HOST'),
        'port': os.environ.get('DB_PORT'),
        'user': os.environ.get('DB_USER'),
        'password': os.environ.get('DB_PASSWORD'),
        'database': os.environ.get('DB_NAME'),
    }
    
    # If all required environment variables are present, use them
    if all([env_vars['host'], env_vars['user'], env_vars['password'], env_vars['database']]):
        # Convert port to int if present, otherwise use default
        if env_vars['port']:
            env_vars['port'] = int(env_vars['port'])
        else:
            env_vars['port'] = 3306
        print("Using database configuration from environment variables")
        return env_vars
    
    # Fall back to YAML file (local development)
    if not os.path.exists(DB_CONFIG_PATH):
        print(f"Error: Database config file not found at {DB_CONFIG_PATH}")
        print("Please either:")
        print("1. Set environment variables: DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME")
        print("2. Create the config file with the following structure:")
        print("""
database:
  host: localhost
  port: 3306
  user: your_username
  password: your_password
  database: your_database_name
""")
        sys.exit(1)

    with open(DB_CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)

    print("Using database configuration from YAML file")
    return config['database']


def connect_to_database(db_config):
    """Establish connection to MySQL database."""
    try:
        connection = mysql.connector.connect(
            host=db_config['host'],
            port=db_config.get('port', 3306),
            user=db_config['user'],
            password=db_config['password'],
            database=db_config['database'],
        )
        connection.cursor().execute("SET time_zone = '+00:00'")
        return connection
    except mysql.connector.Error as e:
        print(f"Error connecting to database: {e}")
        sys.exit(1)


def fetch_view_data(connection, view_name, instrument=None, columns=None):
    """Fetch data from a database view, optionally restricted to one instrument.

    Only EXPORT_COLUMNS are selected: the instrument column is a selection
    criterion only, so the exported CSV keeps exactly the columns it always had.
    """
    if columns is None:
        columns = EXPORT_COLUMNS
    column_list = ", ".join(f"`{col}`" for col in columns)
    query = f"SELECT {column_list} FROM {view_name}"
    params = None
    if instrument is not None:
        query += f" WHERE `{INSTRUMENT_COLUMN}` = %s"
        params = (instrument,)
    try:
        df = pd.read_sql(query, connection, params=params, dtype_backend='numpy_nullable')
        # Keep 'time' as plain string — prevent pandas from auto-parsing ISO 8601
        # strings with timezone offset (e.g. "2026-03-27T00:00:00Z" → local time)
        if 'time' in df.columns:
            df['time'] = df['time'].astype(str)
        if INSTRUMENT_COLUMN in df.columns:
            df = df.drop(columns=[INSTRUMENT_COLUMN])
        return df
    except Exception as e:
        scope = f"' for instrument '{instrument}" if instrument is not None else ""
        print(f"Error fetching data from view '{view_name}{scope}': {e}")
        return None


def export_to_csv(df, output_path):
    """Export DataFrame to CSV file with comma separator."""
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # Normalize any datetime columns to UTC and strip timezone info so CSVs
    # contain unambiguous UTC timestamps without offset suffixes (+02:00 etc.)
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            if df[col].dt.tz is not None:
                df[col] = df[col].dt.tz_convert('UTC').dt.tz_localize(None)
    df.to_csv(output_path, index=False, sep=',')


def commit_and_push_changes(verbose=True):
    """
    Commit and push CSV changes to GitHub repository.
    Only runs in GitHub Actions environment.
    """
    is_github_actions = os.environ.get('GITHUB_ACTIONS') == 'true'
    
    if not is_github_actions:
        if verbose:
            print("Not in GitHub Actions environment, skipping git commit/push")
        return
    
    try:
        import subprocess
        
        # Change to repository root directory
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
        os.chdir(repo_root)
        
        if verbose:
            print(f"Working directory: {os.getcwd()}")
        
        # Configure git
        subprocess.run(['git', 'config', '--global', 'user.name', 'github-actions[bot]'], check=True)
        subprocess.run(['git', 'config', '--global', 'user.email', 'github-actions[bot]@users.noreply.github.com'], check=True)
        
        # Add CSV files individually to avoid glob/path issues
        data_dir = os.path.join('ModelTrading', 'data')
        if os.path.exists(data_dir):
            csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
            for csv_file in csv_files:
                file_path = os.path.join(data_dir, csv_file)
                subprocess.run(['git', 'add', file_path], check=True)
                if verbose:
                    print(f"Added: {file_path}")
        else:
            if verbose:
                print(f"Warning: Data directory not found: {data_dir}")
            return
        
        # Check if there are changes to commit
        result = subprocess.run(['git', 'diff', '--staged', '--quiet'], capture_output=True)
        
        if result.returncode != 0:  # There are changes
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')
            commit_message = f"Auto-update trading data - {timestamp}"
            
            subprocess.run(['git', 'commit', '-m', commit_message], check=True)
            subprocess.run(['git', 'push'], check=True)
            
            if verbose:
                print(f"Successfully committed and pushed changes: {commit_message}")
        else:
            if verbose:
                print("No changes to commit")
                
    except subprocess.CalledProcessError as e:
        print(f"Error during git operations: {e}")
        if verbose:
            print(f"Command: {e.cmd}")
            print(f"Return code: {e.returncode}")
    except Exception as e:
        print(f"Unexpected error during git operations: {e}")


def update_data(exports=None, instruments=None, verbose=True):
    """
    Main function to update data files from database.

    Args:
        exports: List of export configurations. If None, uses DATA_EXPORTS.
        instruments: List of instruments to fetch. If None, uses INSTRUMENTS.
            Every export view is queried once per instrument, and the output file
            name is derived from the instrument (see resolve_output_file).
        verbose: Print progress messages.
    """
    if exports is None:
        exports = DATA_EXPORTS
    if instruments is None:
        instruments = INSTRUMENTS

    if verbose:
        print(f"Starting data update at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("-" * 50)

    # Check if we need to whitelist IP (GitHub Actions environment)
    current_ip = None
    is_github_actions = os.environ.get('GITHUB_ACTIONS') == 'true'
    
    if is_github_actions:
        if verbose:
            print("Running in GitHub Actions environment")
        current_ip = get_current_ip()
        if current_ip:
            if verbose:
                print(f"Current IP address: {current_ip}")
                print("Whitelisting IP for MySQL access...")
            whitelist_ip_cpanel(current_ip)
        else:
            print("Warning: Could not determine current IP address")

    try:
        # Load database configuration
        db_config = load_db_config()

        # Connect to database
        if verbose:
            print(f"Connecting to database at {db_config['host']}...")
        connection = connect_to_database(db_config)

        if verbose:
            print("Connected successfully.")
            print("-" * 50)

        # Process each enabled export, once per instrument
        success_count = 0
        error_count = 0

        active_exports = []
        for export in exports:
            if not export.get('enabled', True):
                if verbose:
                    print(f"Skipping '{export['name']}' (disabled)")
                continue
            active_exports.append(export)

        for instrument in instruments:
            if verbose:
                print("-" * 50)
                print(f"Instrument: {instrument}")

            for export in active_exports:
                view_name = export['view']
                output_file = resolve_output_file(export['output_file'], instrument)
                output_path = os.path.join(DATA_DIR, output_file)

                if verbose:
                    print(f"Processing: {export['name']}")
                    print(f"  View: {view_name}")
                    print(f"  Output: {output_file}")

                # Fetch data from view, restricted to this instrument
                df = fetch_view_data(connection, view_name, instrument=instrument,
                                     columns=export.get('columns'))

                if df is None:
                    error_count += 1
                    continue

                if df.empty:
                    print(f"  WARNING: no rows for '{instrument}' in view '{view_name}' "
                          f"- {output_file} not written")
                    error_count += 1
                    continue

                # Export to CSV
                export_to_csv(df, output_path)

                if verbose:
                    print(f"  Rows exported: {len(df):,}")
                    print(f"  Columns: {', '.join(df.columns)}")

                success_count += 1

        # Close database connection
        connection.close()

        if verbose:
            print("-" * 50)
            print(f"Update complete: {success_count} successful, {error_count} failed")
            print(f"Finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Commit and push changes if in GitHub Actions
        if success_count > 0:
            commit_and_push_changes(verbose=verbose)

        return success_count, error_count
    
    finally:
        # Clean up: Remove IP from whitelist after processing
        if is_github_actions and current_ip:
            if verbose:
                print("Removing IP from whitelist...")
            remove_ip_cpanel(current_ip)


def main():
    """Entry point for command line execution."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Update CSV data files from MySQL database views"
    )
    parser.add_argument(
        '--view',
        type=str,
        help="Only update specific view (by name from DATA_EXPORTS)"
    )
    parser.add_argument(
        '--instrument',
        type=str,
        action='append',
        help="Only update this instrument (e.g. 'EUR/USD' or 'eurusd'). "
             "Repeat the flag for several instruments. Default: all of INSTRUMENTS."
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help="List all configured exports and instruments"
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help="Suppress progress output"
    )

    args = parser.parse_args()

    if args.list:
        print("Configured data exports:")
        print("-" * 50)
        for export in DATA_EXPORTS:
            status = "enabled" if export.get('enabled', True) else "disabled"
            print(f"  {export['name']}: {export['view']} -> {export['output_file']} ({status})")
        print("Configured instruments:")
        print("-" * 50)
        for instrument in INSTRUMENTS:
            print(f"  {instrument} -> {instrument_slug(instrument)}_*.csv")
        return

    # Filter exports if specific view requested
    exports = DATA_EXPORTS
    if args.view:
        exports = [e for e in DATA_EXPORTS if e['name'].lower() == args.view.lower()]
        if not exports:
            print(f"Error: No export found with name '{args.view}'")
            print("Use --list to see available exports")
            sys.exit(1)

    # Filter instruments if specific ones requested
    instruments = INSTRUMENTS
    if args.instrument:
        instruments = []
        for value in args.instrument:
            instrument = normalize_instrument(value)
            if instrument is None:
                print(f"Error: Unknown instrument '{value}'")
                print("Use --list to see available instruments")
                sys.exit(1)
            if instrument not in instruments:
                instruments.append(instrument)

    update_data(exports=exports, instruments=instruments, verbose=not args.quiet)


if __name__ == "__main__":
    main()
