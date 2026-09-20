"""
Fingerprint Feature Configuration Loader

Loads and manages fingerprint-specific feature configuration from fingerprint_features.yaml.
This is separate from the trading model feature configuration.
"""

import yaml
import os


class FingerprintConfig:
    """Load and manage fingerprint feature configuration"""
    
    def __init__(self, config_path=None):
        if config_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            config_path = os.path.join(base_dir, "config", "fingerprint_features.yaml")
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def get_window_size(self):
        """Returns the rolling window size in days"""
        return self.config['window']['size_days']
    
    def get_window_step(self):
        """Returns the rolling window step size in days"""
        return self.config['window']['step_days']
    
    def get_enabled_features(self):
        """Returns list of all enabled feature names"""
        features = []
        for group_name, group_config in self.config['feature_groups'].items():
            if group_config['enabled']:
                features.extend(group_config['features'])
        return features
    
    def get_feature_groups(self):
        """Returns all feature groups configuration"""
        return self.config['feature_groups']
    
    def get_parameters(self):
        """Returns indicator calculation parameters"""
        return self.config['parameters']
    
    def get_rsi_thresholds(self):
        """Returns RSI threshold configuration"""
        return self.config['rsi_thresholds']
    
    def get_similarity_settings(self):
        """Returns similarity analysis settings"""
        return self.config['similarity']
    
    def get_top_n(self):
        """Returns number of top similar windows to return"""
        return self.config['similarity']['top_n']
    
    def get_feature_count(self):
        """Returns total number of enabled features"""
        return len(self.get_enabled_features())


# Singleton instance
_fingerprint_config = None


def get_fingerprint_config(config_path=None):
    """Get or create the singleton fingerprint config instance"""
    global _fingerprint_config
    if _fingerprint_config is None or config_path is not None:
        _fingerprint_config = FingerprintConfig(config_path)
    return _fingerprint_config
