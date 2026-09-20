import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
GENERATED_DIR = os.path.join(BASE_DIR, "generated")
SCENARIOS_DIR = os.path.join(GENERATED_DIR, "scenarios")
REPORT_DIR = os.path.join(BASE_DIR, "report", "python")
SOURCE_DIR = os.path.join(BASE_DIR, "source")
TRAIN_OUTPUT_DIR = os.path.join(BASE_DIR, "training_output")
FEATURE_MAP_DIR = os.path.join(TRAIN_OUTPUT_DIR, "feature_map")
VISUALIZATION_DIR = os.path.join(TRAIN_OUTPUT_DIR, "visualization")
LABEL_REVIEW_DIR = os.path.join(TRAIN_OUTPUT_DIR, "label_review")
CONFIG_DIR = os.path.join(BASE_DIR, "config")
JAVA_CONFIG_DIR = os.path.join(BASE_DIR, SOURCE_DIR, "strategy", "config")


def get_run_dirs(run_id=None):
    """
    Get directory paths for a specific training run.

    When run_id is provided, returns paths to a subdirectory under GENERATED_DIR.
    This allows parallel training runs without file conflicts.

    Args:
        run_id (str, optional): Unique identifier for this training run.
                               If None, uses default directories.

    Returns:
        dict: Dictionary with directory paths:
            - generated_dir: Base directory for generated files
            - report_dir: Directory for backtest reports
            - feature_map_dir: Directory for feature importance maps
            - visualization_dir: Directory for training visualizations
            - label_review_dir: Directory for label distribution analysis
            - java_config_dir: Directory for Java configuration files
    """
    if run_id is None:
        # Default directories (backward compatible)
        return {
            'generated_dir': GENERATED_DIR,
            'report_dir': REPORT_DIR,
            'feature_map_dir': FEATURE_MAP_DIR,
            'visualization_dir': VISUALIZATION_DIR,
            'label_review_dir': LABEL_REVIEW_DIR,
            'java_config_dir': JAVA_CONFIG_DIR
        }
    else:
        # Subdirectories for this specific run
        run_generated_dir = os.path.join(GENERATED_DIR, run_id)
        run_output_dir = os.path.join(run_generated_dir, "training_output")

        return {
            'generated_dir': run_generated_dir,
            'report_dir': os.path.join(run_generated_dir, "report"),
            'feature_map_dir': os.path.join(run_output_dir, "feature_map"),
            'visualization_dir': os.path.join(run_output_dir, "visualization"),
            'label_review_dir': os.path.join(run_output_dir, "label_review"),
            'java_config_dir': os.path.join(run_generated_dir, "config")
        }


def get_scenario_dir(scenario_name):
    """Return the root folder for a named scenario under SCENARIOS_DIR."""
    return os.path.join(SCENARIOS_DIR, scenario_name)