"""
Utility functions for exporting scaler parameters for Java.
"""

import os
import json


def export_scaler_for_java(scaler, feature_names, generated_dir, java_source_dir):
    """
    Export scaler parameters for Java (ScalerParams.java and scaler_config.json).

    Args:
        scaler: Fitted StandardScaler with mean_ and scale_ attributes
        feature_names: List of feature names (must match scaler order)
        generated_dir: Directory to save scaler_config.json
        java_source_dir: Directory to save ScalerParams.java
    """
    print("\n" + "=" * 80)
    print("EXPORTING SCALER PARAMETERS FOR JAVA")
    print("=" * 80 + "\n")

    means = scaler.mean_
    stds = scaler.scale_

    # Export to JSON for easy loading
    scaler_config = {
        "feature_count": len(feature_names),
        "feature_names": list(feature_names),
        "means": means.tolist(),
        "stds": stds.tolist()
    }

    json_path = os.path.join(generated_dir, "scaler_config.json")
    with open(json_path, "w") as f:
        json.dump(scaler_config, f, indent=2)
    print(f"Exported scaler_config.json ({len(feature_names)} features)")

    # Generate ScalerParams.java
    java_path = os.path.join(java_source_dir, "ScalerParams.java")
    os.makedirs(os.path.dirname(java_path), exist_ok=True)

    with open(java_path, "w") as f:
        f.write("package jforex;\n\n")
        f.write("/**\n")
        f.write(" * Auto-generated scaler parameters\n")
        f.write(" * Generated from Python model training\n")
        f.write(" * Uses float arrays for consistency with ONNX Runtime\n")
        f.write(" */\n")
        f.write("public class ScalerParams {\n\n")

        # Feature names
        f.write("    public static final String[] FEATURE_NAMES = {\n")
        for i, name in enumerate(feature_names):
            comma = "," if i < len(feature_names) - 1 else ""
            f.write(f'        "{name}"{comma}\n')
        f.write("    };\n\n")

        # Means
        f.write("    public static final float[] MEANS = {\n")
        for i in range(0, len(means), 5):
            values = means[i:i + 5]
            formatted = ", ".join([f"{v:.10f}f" for v in values])
            comma = "," if i + 5 < len(means) else ""
            f.write(f"        {formatted}{comma}\n")
        f.write("    };\n\n")

        # Stds
        f.write("    public static final float[] STDS = {\n")
        for i in range(0, len(stds), 5):
            values = stds[i:i + 5]
            formatted = ", ".join([f"{v:.10f}f" for v in values])
            comma = "," if i + 5 < len(stds) else ""
            f.write(f"        {formatted}{comma}\n")
        f.write("    };\n\n")

        # Getter methods
        f.write("    public static float[] getMeans() { return MEANS; }\n")
        f.write("    public static float[] getStds() { return STDS; }\n")
        f.write("    public static String[] getFeatureNames() { return FEATURE_NAMES; }\n")
        f.write("}\n")

    print(f"Generated ScalerParams.java ({len(feature_names)} features)")
    print("OK: Scaler export complete")


def _write_java_array(f, name, values, is_float=True):
    """Helper to write a Java array declaration."""
    if is_float:
        f.write(f"    public static final float[] {name} = {{\n")
        for i in range(0, len(values), 5):
            chunk = values[i:i + 5]
            formatted = ", ".join([f"{v:.10f}f" for v in chunk])
            comma = "," if i + 5 < len(values) else ""
            f.write(f"        {formatted}{comma}\n")
        f.write("    };\n\n")
    else:
        f.write(f"    public static final String[] {name} = {{\n")
        for i, val in enumerate(values):
            comma = "," if i < len(values) - 1 else ""
            f.write(f'        "{val}"{comma}\n')
        f.write("    };\n\n")


def export_scalers_for_java(scalers_by_model, generated_dir, java_source_dir):
    """Export per-model scaler parameters for Java.

    Args:
        scalers_by_model: dict mapping model key (long_fast, short_fast, long_slow,
            short_slow, reg) -> (fitted StandardScaler, feature_names list)
        generated_dir: dir for scaler_config.json
        java_source_dir: dir for ScalerParams.java

    Generates ScalerParams.java with per-model <MODEL>_FEATURE_NAMES/_MEANS/_STDS
    arrays plus model-generic accessors getMeans(model)/getStds(model)/
    getFeatureNames(model), and scaler_config.json keyed by model.
    """
    print("\n" + "=" * 80)
    print("EXPORTING PER-MODEL SCALER PARAMETERS FOR JAVA")
    print("=" * 80 + "\n")

    models = list(scalers_by_model.keys())

    # Export to JSON (one object per model)
    scaler_config = {}
    for mk, (sc, feats) in scalers_by_model.items():
        scaler_config[mk] = {
            "feature_count": len(feats),
            "feature_names": list(feats),
            "means": sc.mean_.tolist(),
            "stds": sc.scale_.tolist(),
        }

    json_path = os.path.join(generated_dir, "scaler_config.json")
    with open(json_path, "w") as f:
        json.dump(scaler_config, f, indent=2)
    print("Exported scaler_config.json (" +
          ", ".join(f"{mk}: {len(feats)}" for mk, (sc, feats) in scalers_by_model.items()) + ")")

    # Generate ScalerParams.java
    java_path = os.path.join(java_source_dir, "ScalerParams.java")
    os.makedirs(os.path.dirname(java_path), exist_ok=True)

    with open(java_path, "w") as f:
        f.write("package jforex;\n\n")
        f.write("import java.util.HashMap;\n")
        f.write("import java.util.Map;\n\n")
        f.write("/**\n")
        f.write(" * Auto-generated per-model scaler parameters.\n")
        f.write(" * Each model has its own independent feature set / scaler.\n")
        f.write(" */\n")
        f.write("public class ScalerParams {\n\n")

        for mk, (sc, feats) in scalers_by_model.items():
            prefix = mk.upper()
            f.write(f"    // === {mk} ===\n\n")
            _write_java_array(f, f"{prefix}_FEATURE_NAMES", feats, is_float=False)
            _write_java_array(f, f"{prefix}_MEANS", sc.mean_)
            _write_java_array(f, f"{prefix}_STDS", sc.scale_)

        # Model-generic lookup maps
        f.write("    private static final Map<String, float[]> MEANS_BY_MODEL = new HashMap<>();\n")
        f.write("    private static final Map<String, float[]> STDS_BY_MODEL = new HashMap<>();\n")
        f.write("    private static final Map<String, String[]> NAMES_BY_MODEL = new HashMap<>();\n")
        f.write("    static {\n")
        for mk in models:
            prefix = mk.upper()
            f.write(f'        MEANS_BY_MODEL.put("{mk}", {prefix}_MEANS);\n')
            f.write(f'        STDS_BY_MODEL.put("{mk}", {prefix}_STDS);\n')
            f.write(f'        NAMES_BY_MODEL.put("{mk}", {prefix}_FEATURE_NAMES);\n')
        f.write("    }\n\n")

        f.write("    public static float[] getMeans(String model) { return MEANS_BY_MODEL.get(model); }\n")
        f.write("    public static float[] getStds(String model) { return STDS_BY_MODEL.get(model); }\n")
        f.write("    public static String[] getFeatureNames(String model) { return NAMES_BY_MODEL.get(model); }\n")
        f.write("}\n")

    print("Generated ScalerParams.java (" +
          ", ".join(f"{mk}: {len(feats)}" for mk, (sc, feats) in scalers_by_model.items()) + ")")
    print("OK: Per-model scaler export complete")
