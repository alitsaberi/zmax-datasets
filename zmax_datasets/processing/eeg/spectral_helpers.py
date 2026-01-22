# -*- coding: utf-8 -*-
"""
Created on Thu Jan 22 14:26:08 2026

@author: selaca
"""

from __future__ import annotations

import numpy as np
from scipy.signal import welch


def integrate_bandpower(psd_lin, freqs, fmin, fmax):
    """Integrate linear PSD over a frequency band.

   Computes band power by integrating the power spectral density (PSD) between
   `fmin` (inclusive) and `fmax` (exclusive) using the trapezoidal rule.

   Args:
       psd_lin: Linear PSD values with shape (..., n_freqs). The integration is
           performed along the last axis.
       freqs: Frequency vector of shape (n_freqs,).
       fmin: Lower frequency bound in Hz (inclusive).
       fmax: Upper frequency bound in Hz (exclusive).

   Returns:
       Band power with shape psd_lin.shape[:-1]. If no frequencies fall in the
       requested band, returns zeros with the appropriate shape.
   """
    mask = (freqs >= fmin) & (freqs < fmax)
    if not np.any(mask):
        # Return zeros with correct leading shape if band not present
        return np.zeros(psd_lin.shape[:-1], dtype=float)
    return np.trapz(psd_lin[..., mask], freqs[mask], axis=-1)

def welch_psd(
    x: np.ndarray,
    *,
    sf: float,
    nperseg: int,
    noverlap: int,
    axis: int = -1,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Welch PSD.

    Args:
        x: Time series array.
        sf: Sampling frequency in Hz.
        nperseg: Segment length in samples.
        noverlap: Overlap length in samples.
        axis: Axis corresponding to time.

    Returns:
        Tuple (psd, freqs) where:
            - psd has the same shape as `x` except time axis replaced by n_freqs
            - freqs has shape (n_freqs,)
    """
    freqs, psd = welch(x, fs=sf, nperseg=nperseg, noverlap=noverlap, axis=axis)
    return psd, freqs
