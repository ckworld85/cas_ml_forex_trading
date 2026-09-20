"""
Standalone script to export scaler parameters for Java.

This script loads a trained scaler and exports it to:
- scaler_config.json (JSON format)
- ScalerParams.java (Java class with hardcoded arrays)

Note: train.py and advanced_train.py now call export_scaler automatically,
so this script is only needed for manual re-export.
"""

import os
import sys
import joblib
import pandas as pd

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import ModelTrading.config.directories as dir_config
from ModelTrading.source.python.export.export_scaler import export_scaler_for_java

# Configuration
GENERATED_DIR = dir_config.GENERATED_DIR
JAVA_SOURCE_DIR = os.path.join(dir_config.SOURCE_DIR, "strategy", "src", "main", "java", "jforex")

# Load trained components
print("Loading scaler and feature data...")
scaler = joblib.load(os.path.join(GENERATED_DIR, "scaler.save"))
X = pd.read_parquet(os.path.join(GENERATED_DIR, "X.parquet"))
feature_names = list(X.columns)

# Export using shared utility function
export_scaler_for_java(scaler, feature_names, GENERATED_DIR, JAVA_SOURCE_DIR)

# Print summary
means = scaler.mean_
stds = scaler.scale_

print(f"\n{'=' * 70}")
print("SUMMARY")
print(f"{'=' * 70}")
print(f"Features: {len(feature_names)}")
print(f"Data type: float (32-bit)")
print(f"\nFirst 5 features:")
for i, name in enumerate(feature_names[:5]):
    print(f"  {i + 1}. {name:25s} mean={means[i]:10.6f}  std={stds[i]:10.6f}")
if len(feature_names) > 5:
    print(f"  ... and {len(feature_names) - 5} more")
print(f"\n{'=' * 70}")
print("\nNext steps:")
print("1. Copy strategy_model_*.onnx to JForex directory")
print("2. Compile ScalerParams.java in your JForex project")
print("3. Run the strategy")
print(f"{'=' * 70}\n")
