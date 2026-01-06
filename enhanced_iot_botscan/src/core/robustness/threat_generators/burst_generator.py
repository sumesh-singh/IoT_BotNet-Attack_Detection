"""
Burst Generator - Enhanced IoT BotScan
Author: Kotiwale Sumesh Singh (160124862043)

Simulates burst traffic patterns and DDoS-like conditions.
"""

import numpy as np
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class BurstGenerator:
    """Simulate various burst traffic patterns."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize burst generator."""
        self.config = config or {}
        self.seed = self.config.get('random_seed', 42)
        np.random.seed(self.seed)
        
        logger.info("BurstGenerator initialized")
    
    def simulate_burst_traffic(self, X: np.ndarray, intensity: float = 1.5) -> np.ndarray:
        """
        Simulate traffic burst by amplifying feature values.
        
        Args:
            X: Input data (n_samples, n_features)
            intensity: Burst intensity multiplier (1.0 = no burst)
            
        Returns:
            Data with burst traffic simulation
        """
        if intensity == 1.0:
            return X.copy()
        
        X_burst = X * intensity
        
        # Clip to reasonable range (features shouldn't exceed 10x normal range)
        feature_maxs = np.max(X, axis=0) * 3
        X_burst = np.clip(X_burst, np.min(X, axis=0), feature_maxs)
        
        logger.debug(f"Simulated burst traffic with intensity {intensity}")
        return X_burst
    
    def simulate_ddos_pattern(self, X: np.ndarray, attack_rate: float = 0.3,
                             amplification: float = 5.0) -> np.ndarray:
        """
        Simulate DDoS attack pattern (sudden spike in certain features).
        
        Args:
            X: Input data
            attack_rate: Fraction of samples affected
            amplification: Amplification factor for affected samples
            
        Returns:
            Data with DDoS pattern
        """
        X_ddos = X.copy()
        n_samples = X.shape[0]
        n_affected = int(n_samples * attack_rate)
        
        # Select random samples to affect
        affected_indices = np.random.choice(n_samples, n_affected, replace=False)
        
        # Amplify network-related features (assuming first 20% of features are network-related)
        n_features = X.shape[1]
        network_features = int(n_features * 0.2)
        
        X_ddos[affected_indices, :network_features] *= amplification
        
        # Clip to valid range
        feature_maxs = np.max(X, axis=0) * 5
        X_ddos = np.clip(X_ddos, np.min(X, axis=0), feature_maxs)
        
        logger.debug(f"Simulated DDoS pattern affecting {n_affected} samples")
        return X_ddos
    
    def simulate_flash_crowd(self, X: np.ndarray, crowd_size: int = 100,
                            burst_duration: int = 50) -> np.ndarray:
        """
        Simulate flash crowd event (sudden legitimate traffic spike).
        
        Args:
            X: Input data
            crowd_size: Number of burst samples
            burst_duration: Duration of burst in samples
            
        Returns:
            Data with flash crowd pattern
        """
        X_crowd = X.copy()
        n_samples = X.shape[0]
        
        # Select random start point for burst
        burst_start = np.random.randint(0, max(1, n_samples - burst_duration))
        burst_end = min(burst_start + burst_duration, n_samples)
        
        # Amplify traffic in burst window
        X_crowd[burst_start:burst_end] *= 2.0
        
        # Clip to valid range
        feature_maxs = np.max(X, axis=0) * 3
        X_crowd = np.clip(X_crowd, np.min(X, axis=0), feature_maxs)
        
        logger.debug(f"Simulated flash crowd from sample {burst_start} to {burst_end}")
        return X_crowd
    
    def simulate_port_scan(self, X: np.ndarray, scan_rate: float = 0.2) -> np.ndarray:
        """
        Simulate port scanning pattern (many connections to different ports).
        
        Args:
            X: Input data
            scan_rate: Fraction of samples affected
            
        Returns:
            Data with port scan pattern
        """
        X_scan = X.copy()
        n_samples = X.shape[0]
        n_features = X.shape[1]
        n_affected = int(n_samples * scan_rate)
        
        # Select samples for port scan
        scan_indices = np.random.choice(n_samples, n_affected, replace=False)
        
        # Increase connection diversity features (assuming middle 20% of features)
        diversity_features = slice(int(n_features * 0.4), int(n_features * 0.6))
        
        X_scan[scan_indices, diversity_features] *= 3.0
        
        # Clip to valid range
        feature_maxs = np.max(X, axis=0) * 4
        X_scan = np.clip(X_scan, np.min(X, axis=0), feature_maxs)
        
        logger.debug(f"Simulated port scan affecting {n_affected} samples")
        return X_scan
    
    def simulate_pulse_wave(self, X: np.ndarray, pulse_frequency: int = 20,
                           pulse_amplitude: float = 2.0) -> np.ndarray:
        """
        Simulate pulsed attack pattern (periodic bursts).
        
        Args:
            X: Input data
            pulse_frequency: Samples between pulses
            pulse_amplitude: Amplification during pulse
            
        Returns:
            Data with pulse wave pattern
        """
        X_pulse = X.copy()
        n_samples = X.shape[0]
        
        # Create pulse mask
        pulse_mask = np.zeros(n_samples, dtype=bool)
        pulse_mask[::pulse_frequency] = True
        
        # Apply pulses
        X_pulse[pulse_mask] *= pulse_amplitude
        
        # Clip to valid range
        feature_maxs = np.max(X, axis=0) * 3
        X_pulse = np.clip(X_pulse, np.min(X, axis=0), feature_maxs)
        
        logger.debug(f"Simulated pulse wave with frequency {pulse_frequency}")
        return X_pulse
    
    def simulate_asymmetric_traffic(self, X: np.ndarray, asymmetry_ratio: float = 5.0) -> np.ndarray:
        """
        Simulate asymmetric traffic (e.g., large downloads, small uploads).
        
        Args:
            X: Input data
            asymmetry_ratio: Ratio between upload and download features
            
        Returns:
            Data with asymmetric pattern
        """
        X_asym = X.copy()
        n_features = X.shape[1]
        
        # Assume first half are download-related, second half upload-related
        download_features = slice(0, n_features // 2)
        upload_features = slice(n_features // 2, n_features)
        
        X_asym[:, download_features] *= asymmetry_ratio
        X_asym[:, upload_features] /= asymmetry_ratio
        
        # Clip to valid range
        feature_maxs = np.max(X, axis=0) * 10
        feature_mins = np.min(X, axis=0) / 10
        X_asym = np.clip(X_asym, feature_mins, feature_maxs)
        
        logger.debug(f"Simulated asymmetric traffic with ratio {asymmetry_ratio}")
        return X_asym
