

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from zmax_datasets.processing.eeg.artifact_detection.apply_and_label import (
    detect_artifacts,
)
from zmax_datasets.processing.eeg.artifact_detection.features import (
    epoch_features_from_single_channel_epochs,
)
from zmax_datasets.processing.eeg.artifact_detection.preprocessing import (
    create_and_process_mne,
)
from zmax_datasets.utils.data import Data


def run_full_zmax_artifact_pipeline_from_data(
    data: Data,
) -> dict:
    
    """Run the full rule-based artifact labeling pipeline on a Data object.

    This is a wrapper that converts a `Data` object into the EEG dataframe format,
    runs per-channel preprocessing (MNE), extracts epoch-level features, derives
    per-feature thresholds, and produces a final per-epoch artifact label for
    each available channel.

    If only one EEG channel is available, the missing channel output is returned
    as NaNs.

    Args:
        data: Input time-series data.
      

    Returns:
        Dictionary with:
        - bad_indices: Per-epoch labels for left channel (float array; NaN if missing).
    """

   
    sample_rate = float(data.sample_rate)
    channel_name= data.channel_names[0]
    eeg = pd.DataFrame({
    'timestamp':data.timestamps,
    channel_name: data.array[:, 0]
    })

    ################## CREATE AND PROCESS MNE RAW ####################
    processed_outputs = create_and_process_mne(
        eeg, sample_rate, channel_name=channel_name
    )
    logger.debug("Available channels: {}", list(processed_outputs.keys()))

    if processed_outputs:
        ################## EXTRACT EPOCH FEATURES ########################
        df, freqs, psd = epoch_features_from_single_channel_epochs(
            processed_outputs["epochs_phys"], channel_name=channel_name
        )
        ############## THRESHOLD DISCOVERY ############################
        dfp, info = detect_artifacts(df)
        ############### BUILD OUTPUT ARRAYS (NaNs for missing channel) ###############
        
        bad_indices = dfp.sort_values("epoch")["bad_independent"].to_numpy(dtype=float)
        return {"artifact_epochs": bad_indices, "info": info}
    else:
        return {
            "artifact_epochs": np.zeros(0, dtype=float),
            "info": np.zeros(0, dtype=float),
        }


