"""
adaptive_learning.py
=====================
Was an empty (0-byte) stub in the original scaffold.

Scope, stated honestly: this recalibrates the Autoencoder's anomaly
THRESHOLD over time using a rolling window of recent flows that other
detectors did NOT flag (a proxy for "probably normal" live traffic) --
it does not retrain the RF, Autoencoder, or GAT weights themselves.
Concept drift in real network traffic (new devices, new normal
behavior) will change what "normal" reconstruction error looks like;
periodically recalibrating the threshold against recent traffic is a
real, defensible way to track that without needing new labels. Full
online retraining of the neural weights is future work, not something
to claim here without building and validating it.
"""
from collections import deque


class AdaptiveThresholdManager:
    def __init__(self, initial_threshold, window_size=500,
                 percentile=99.0, min_samples=100):
        self.threshold = initial_threshold
        self.window_size = window_size
        self.percentile = percentile
        self.min_samples = min_samples
        self._recent_normal_errors = deque(maxlen=window_size)
        self.recalibration_count = 0

    def observe(self, ae_mse, was_flagged_by_other_detectors):
        """Feed in each flow's AE reconstruction error. Only errors from
        flows NOT flagged by RF/GAT are used to update the baseline --
        using flagged flows would let an ongoing attack poison the
        'normal' baseline upward and blind the detector to itself."""
        if not was_flagged_by_other_detectors:
            self._recent_normal_errors.append(ae_mse)

        if len(self._recent_normal_errors) >= self.min_samples:
            self._recalibrate()

    def _recalibrate(self):
        import numpy as np
        new_threshold = float(np.percentile(self._recent_normal_errors, self.percentile))
        # damped update -- avoid the threshold swinging wildly on a small window
        self.threshold = 0.7 * self.threshold + 0.3 * new_threshold
        self.recalibration_count += 1

    def get_threshold(self):
        return self.threshold
