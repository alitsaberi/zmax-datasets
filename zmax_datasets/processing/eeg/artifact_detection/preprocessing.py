"""
Created on Wed Feb 11 12:27:42 2026

@author: selaca
"""

import mne
import numpy as np
import pandas as pd

from zmax_datasets.processing.eeg.artifact_detection.constants import (
    EPOCH_DURATION,
    NOTCH_FREQ,
    PHYS_BAND,
    VLF_BAND,
)
from zmax_datasets.processing.eeg.utils import (
    ensure_volts,
)
from zmax_datasets.utils.exceptions import (
    EmptyDataFrameError,
    MissingColumnError,
)


def create_and_process_mne(eeg: pd.DataFrame, sample_rate: float, channel_name):
    """Create MNE objects and epochs for available EEG channels.

    Processes left and right EEG channels independently if present in the input
    dataframe. Channels that are missing or all-NaN are skipped.

    Args:
        eeg: Timeseries dataframe containing `timestamp` and optionally "EEG L"
            and/or "EEG R".
        sample_rate: Sampling rate in Hz.

    Returns:
        Dictionary mapping channel name ("EEG_L", "EEG_R") to per-channel outputs
        produced by `process_eeg_channel`.

    Raises:
        ValueError: If neither channel is available (missing or all-NaN).
    """
    out = process_eeg_channel(
         eeg,
        channel_name=channel_name,
        sample_rate=sample_rate,
    )
    
    return out



def process_eeg_channel(
    eeg: pd.DataFrame,
    *,
    channel_name: str,
    sample_rate: float,
    epoch_duration: float = EPOCH_DURATION,
    notch_freq: float = NOTCH_FREQ,
    phys_band: tuple[float, float] = PHYS_BAND,
    vlf_band: tuple[float, float] = VLF_BAND,
    ):
    """Create an MNE Raw object for one EEG channel and run phys/VLF pipelines.

    Args:
        eeg: Timeseries dataframe containing `channel_column` and `timestamp`.
        channel_column: Column name in `eeg` (e.g. "EEG L").
        channel_name: MNE channel name to use (e.g. "EEG_L").
        sample_rate: Sampling rate in Hz.
        epoch_duration: Epoch duration in seconds (fixed-length epochs).
        notch_freq: Notch filter frequency in Hz (e.g. 50 Hz).
        phys_band: Bandpass (low, high) for phys pipeline.
        vlf_band: Bandpass (low, high) for VLF pipeline.

    A dictionary with:
            - channel: The channel name used in MNE.
            - raw: Base Raw object (notch-filtered).
            - raw_phys: Raw object filtered with `phys_band`.
            - raw_vlf: Raw object filtered with `vlf_band`.
            - epochs_phys: Fixed-length epochs from `raw_phys`.
            - phys_signal: 1D NumPy array of phys-filtered signal.
            - vlf_signal: 1D NumPy array of VLF-filtered signal.
            - epochs_data: 2D NumPy array of shape (n_epochs, n_times).

    Raises:
        KeyError: If required columns are missing.
        ValueError: If timestamps are invalid or data is empty.
    """
    if channel_name not in eeg.columns:
       raise MissingColumnError(channel_name)
    if "timestamp" not in eeg.columns:
        raise MissingColumnError("timestamp")
    if len(eeg) == 0:
        raise EmptyDataFrameError()

    # Convert signal to numpy and ensure it's in volts.
    signal = eeg[channel_name].to_numpy()
    # All NaN -> skip (channel not available)
    if not np.isfinite(signal).any():
        return None
    
    
    signal_v = ensure_volts(signal)
    
    # If ensure_volts returns NaNs only (extremely defensive)
    if not np.isfinite(signal_v).any():
        return None

    info = mne.create_info([channel_name], sfreq=sample_rate, ch_types="eeg")
    raw = mne.io.RawArray(signal_v[np.newaxis, :], info)

    # Notch filter (applied before branching into pipelines)
    raw.notch_filter(notch_freq)

    # --- PHYS pipeline (artifact detection) ---
    raw_phys = raw.copy()
    
    raw_phys.filter(
    phys_band[0],
    phys_band[1],
    phase="zero-double",
    fir_design="firwin",
    )

    # --- VLF pipeline (sweat audit) ---
    raw_vlf = raw.copy()
    
    raw_vlf.filter(
    vlf_band[0],
    vlf_band[1],
    phase="zero-double",
    fir_design="firwin",
    )
    
    # Epochs from phys pipeline
    epochs_phys = mne.make_fixed_length_epochs(
    raw_phys,
    duration=epoch_duration,
    preload=True,
    )

    # Extract 1D filtered signals
    phys_signal = raw_phys.get_data(picks=[channel_name])[0]
    vlf_signal = raw_vlf.get_data(picks=[channel_name])[0]

    # Extract epochs data (n_epochs, n_channels=1, n_times) -> (n_epochs, n_times)
    epochs_data = epochs_phys.get_data(picks=[channel_name])[:, 0, :]

    return {
        'channel': channel_name,
        'raw': raw,
        'raw_phys': raw_phys,
        'raw_vlf': raw_vlf,
        'epochs_phys': epochs_phys,
        'phys_signal': phys_signal,
        'vlf_signal': vlf_signal,
        'epochs_data': epochs_data
        }



