# feature_extraction.py
import numpy as np
from scipy.fftpack import dct

class FeatureExtraction:

    def __init__(self, ac_keep=50, dct_keep=20):
        """
        ac_keep  -> number of significant autocorrelation coefficients to keep
        dct_keep -> number of strongest DCT coefficients to keep
        """
        self.ac_keep = ac_keep
        self.dct_keep = dct_keep

    # -----------------------------------------------------------
    # 1) AUTOCORRELATION (AC)
    # -----------------------------------------------------------
    def autocorrelation(self, signal):
        """
        Computes full autocorrelation of a 1D ECG signal.
        Keeps only the strongest 'ac_keep' coefficients.
        """
        # Full AC (numpy correlates with reversed array)
        ac_full = np.correlate(signal, signal, mode='full')

        # Keep center part (positive lags)
        ac = ac_full[ac_full.size // 2:]

        # Select significant AC values based on magnitude
        if len(ac) > self.ac_keep:
            idx = np.argsort(np.abs(ac))[::-1][:self.ac_keep]
            ac = ac[idx]
            # Sort back by lag order
            ac = ac[np.argsort(idx)]

        return ac

    # -----------------------------------------------------------
    # 2) DCT (Discrete Cosine Transform)
    # -----------------------------------------------------------
    def apply_dct(self, ac_coeffs):
        """
        Apply DCT on selected AC coefficients.
        Keep the non-zero / strongest dct_keep coefficients.
        """
        dct_values = dct(ac_coeffs, norm='ortho')

        # Select strongest DCT coefficients
        if len(dct_values) > self.dct_keep:
            idx = np.argsort(np.abs(dct_values))[::-1][:self.dct_keep]
            dct_values = dct_values[idx]
            dct_values = dct_values[np.argsort(idx)]

        return dct_values

    # -----------------------------------------------------------
    # 3) FULL FEATURE EXTRACTION PIPELINE
    # -----------------------------------------------------------
    def extract_features(self, signal):
        """
        Step 2 complete:
        - Compute autocorrelation
        - Get significant coefficients
        - Apply DCT
        - Return DCT feature vector
        """

        ac = self.autocorrelation(signal)
        dct_feats = self.apply_dct(ac)

        return {
            "autocorrelation": ac,
            "dct_features": dct_feats
        }
