from enum import Enum

import neurokit2 as nk
import numpy as np
import pandas as pd

from zmax_datasets.transforms.base import Transform
from zmax_datasets.utils.data import Data


def get_peak_indices(array: np.ndarray) -> np.ndarray:
    return np.where(array == 1)[0]


class FixPeaks(Transform):
    """
    Extract artifacts from peaks and correct them.
    """

    FIXED_PEAKS_CHANNEL_NAME = "fixed_peaks"
    ARTIFACTUAL_PEAKS_CHANNEL_NAME = "artifactual_peaks"
    ARTIFACTUAL_PEAK_TYPES = [
        "ectopic_peaks",
        "missed_peaks",
        "extra_peaks",
        "longshort_peaks",
    ]

    def __init__(
        self,
        aggregate_artifacts: bool = False,
    ):
        """
        Args:
            aggregate_artifacts (bool): If True, sum all artifact channels
                (ectopic_peaks, missed_peaks, extra_peaks, longshort_peaks)
                into a single signal.
        """
        self.aggregate_artifacts = aggregate_artifacts

    def __call__(self, data: Data) -> Data:
        if data.n_channels != 1:
            raise ValueError(
                "ECGFixPeaks expects 1 channel (peaks)."
                f" Found {data.n_channels} channels."
            )

        peak_indices = get_peak_indices(data.array)

        info, fixed_peaks = nk.signal_fixpeaks(
            peak_indices, sampling_rate=int(data.sample_rate)
        )

        info = {
            "fixed_Peaks": fixed_peaks,
            "ectopic_Peaks": info["ectopic"],
            "missed_Peaks": info["missed"],
            "extra_Peaks": info["extra"],
            "longshort_Peaks": info["longshort"],
        }

        signals = nk.signal_formatpeaks(
            info,
            desired_length=data.length,
            peak_indices=fixed_peaks,
        )

        array = signals.values
        channel_names = [self.FIXED_PEAKS_CHANNEL_NAME] + self.ARTIFACTUAL_PEAK_TYPES

        if self.aggregate_artifacts:
            # Sum all artifact channels (columns 1-4) into one
            artifact_signal = array[:, 1:].sum(axis=1)
            array = np.column_stack([array[:, 0], artifact_signal])
            channel_names = [
                self.FIXED_PEAKS_CHANNEL_NAME,
                self.ARTIFACTUAL_PEAKS_CHANNEL_NAME,
            ]

        return Data(
            array=array,
            sample_rate=data.sample_rate,
            timestamps=data.timestamps,
            channel_names=channel_names,
        )


class HeartRate(Transform):
    """
    Calculate heart rate from ECG peaks.
    """

    CHANNEL_NAMES = ["ibi", "hr"]

    class InterpolationMethod(Enum):
        LINEAR = "linear"
        NEAREST = "nearest"
        ZERO = "zero"
        SLINEAR = "slinear"
        QUADRATIC = "quadratic"
        CUBIC = "cubic"
        PREVIOUS = "previous"
        NEXT = "next"
        MONOTONE_CUBIC = "monotone_cubic"
        AKIMA = "akima"

    def __init__(self, interpolation_method: str = InterpolationMethod.MONOTONE_CUBIC):
        self.interpolation_method = interpolation_method

    def __call__(self, data: Data) -> Data:
        if data.n_channels != 1:
            raise ValueError(
                "HeartRate expects 1 channel (peaks)."
                f" Found {data.n_channels} channels."
            )

        peaks = get_peak_indices(data.array)

        periods = nk.signal_period(
            peaks,
            sampling_rate=int(data.sample_rate),
            desired_length=data.length,
            interpolation_method=self.interpolation_method,
        )

        ibi = periods * 1000
        rate = 60 / periods

        array = np.array([ibi, rate]).T

        return Data(
            array=array,
            sample_rate=data.sample_rate,
            timestamps=data.timestamps,
            channel_names=self.CHANNEL_NAMES,
        )


class HRVTime(Transform):
    """
    HRV transform that computes HRV metrics for overlapping sliding windows.

    This version maintains the exact same output frequency as the input signal
    by computing HRV metrics for each sample position using a look-back sliding window.
    Each window contains data from the current position looking backwards in time.

    Args:
        window_size: Size of the sliding window in seconds
        hrv_indices: List of HRV indices to extract (default: None - extracts all)
        min_peaks: Minimum number of peaks required in window to compute HRV
        fill_method: Method to handle windows with insufficient peaks
            ('nan', 'interpolate', 'last_valid')
        output_sample_rate: Sample rate for the output HRV data (default: 4.0 Hz)
    """

    WINDOW_SIZE = 30.0
    INDICES = [
        "HRV_SDNN",
        "HRV_RMSSD",
    ]
    FILL_METHODS = ["nan", "interpolate", "last_valid"]

    def __init__(
        self,
        window_size: float = 30.0,
        hrv_indices: list[str] | None = None,
        min_peaks: int = 3,
        fill_method: str = "nan",
        output_sample_rate: float = 4.0,
    ):
        self.window_size = window_size
        self.hrv_indices = hrv_indices
        self.min_peaks = min_peaks
        self.output_sample_rate = output_sample_rate

        if fill_method not in self.FILL_METHODS:
            raise ValueError(
                f"Invalid fill_method: {fill_method}. "
                f"Must be one of: {self.FILL_METHODS}"
            )

        self.fill_method = fill_method

        # Validate HRV indices if provided
        if self.hrv_indices is not None:
            invalid_indices = list(set(self.hrv_indices) - set(self.INDICES))
            if invalid_indices:
                raise ValueError(
                    f"Invalid HRV indices: {invalid_indices}. "
                    f"Available indices: {self.INDICES}"
                )

    def __call__(self, data: Data) -> Data:
        """
        Apply HRV sliding window transform to the input data.

        Args:
            data: Data object containing binary peak signal

        Returns:
            Data object containing HRV metrics with same frequency as input
        """
        if data.array.shape[1] != 1:
            raise ValueError(f"Expected 1 channel, got {data.array.shape[1]}")

        if data.duration.total_seconds() < self.window_size:
            raise ValueError(
                f"Duration of data is less than window size: "
                f"{data.duration.total_seconds()} < {self.window_size}"
            )

        peak_signal = data.array.squeeze()

        window_samples = int(self.window_size * data.sample_rate)

        # Calculate step size to achieve desired output sample rate
        step_size = int(data.sample_rate / self.output_sample_rate)

        hrv_metrics_list = []

        for i in range(0, len(peak_signal), step_size):
            # Define window boundaries - look back from current position
            window_start = max(0, i - window_samples + 1)
            window_end = i + 1

            window_peaks = peak_signal[window_start:window_end]

            # Convert binary peaks to peak indices within the window
            peak_indices = np.where(window_peaks == 1)[0]

            # Compute HRV metrics for this window
            hrv_metrics = self._compute_hrv_metrics(peak_indices, data.sample_rate)
            hrv_metrics_list.append(hrv_metrics)

        hrv_df = pd.DataFrame(hrv_metrics_list)

        # Handle filling for windows with insufficient peaks
        hrv_df = self._fill_insufficient_windows(hrv_df)

        # Select specific HRV indices if requested
        if self.hrv_indices is not None:
            available_cols = [col for col in self.hrv_indices if col in hrv_df.columns]
            hrv_df = hrv_df[available_cols]

        # Convert to numpy array and ensure 2D
        hrv_array = hrv_df.values
        if hrv_array.ndim == 1:
            hrv_array = hrv_array.reshape(-1, 1)

        # Create output Data object with specified output sample rate
        return Data(
            array=hrv_array,
            sample_rate=self.output_sample_rate,
            channel_names=list(hrv_df.columns),
            timestamp_offset=data.timestamps[0] if hasattr(data, "timestamps") else 0,
        )

    def _compute_hrv_metrics(
        self, peak_indices: np.ndarray, sample_rate: float
    ) -> dict[str, float]:
        """
        Compute HRV metrics for a given set of peak indices.

        Args:
            peak_indices: Array of peak indices within the window
            sample_rate: Sampling rate of the original signal

        Returns:
            Dictionary containing HRV metrics
        """
        if len(peak_indices) < self.min_peaks:
            # Return NaN values for all metrics if insufficient peaks
            return {idx: np.nan for idx in self.INDICES}

        try:
            # Convert peak indices to R-R intervals (in milliseconds)
            rr_intervals = np.diff(peak_indices) * (1000.0 / sample_rate)

            # Calculate only RMSSD and SDNN manually
            hrv_metrics = {}

            # Standard deviation of NN intervals (SDNN)
            hrv_metrics["HRV_SDNN"] = np.std(rr_intervals, ddof=1)

            # Root mean square of successive differences (RMSSD)
            if len(rr_intervals) > 1:
                successive_diffs = np.diff(rr_intervals)
                hrv_metrics["HRV_RMSSD"] = np.sqrt(np.mean(successive_diffs**2))
            else:
                hrv_metrics["HRV_RMSSD"] = 0.0

            return hrv_metrics

        except Exception:
            # Return NaN values if computation fails
            return {idx: np.nan for idx in self.INDICES}

    def _fill_insufficient_windows(self, hrv_df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle windows with insufficient peaks based on fill_method.

        Args:
            hrv_df: DataFrame containing HRV metrics

        Returns:
            DataFrame with filled values
        """
        if self.fill_method == "nan":
            return hrv_df  # Already contains NaNs

        elif self.fill_method == "interpolate":
            return hrv_df.interpolate(method="linear", limit_direction="both")

        elif self.fill_method == "last_valid":
            return hrv_df.fillna(method="ffill").fillna(method="bfill")

        return hrv_df
