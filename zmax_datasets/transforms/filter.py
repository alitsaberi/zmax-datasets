from mne.filter import filter_data

from zmax_datasets.transforms.base import Transform
from zmax_datasets.utils.data import Data


class FIRFilter(Transform):
    """
    Apply FIR filter with a Hamming window. The filter length is automatically
    chosen using the filter design function.

    If both `low_cutoff` and `high_cutoff` are None,
    the original data is returned unchanged.
    If only `low_cutoff` is provided, a highpass filter is applied.
    If only `high_cutoff` is provided, a lowpass filter is applied.
    If `low_cutoff` is greater than `high_cutoff`, a bandstop filter is applied.
    If `low_cutoff` is less than `high_cutoff`, a bandpass filter is applied.
    """

    def __init__(
        self,
        low_cutoff: float | None = None,
        high_cutoff: float | None = None,
    ):
        self.low_cutoff = low_cutoff
        self.high_cutoff = high_cutoff

    def __call__(
        self,
        data: Data,
        **kwargs,
    ) -> Data:
        return Data(
            filter_data(
                data.array.T,
                data.sample_rate,
                self.low_cutoff,
                self.high_cutoff,
                **kwargs,
            ).T,
            sample_rate=data.sample_rate,
            channel_names=data.channel_names,
            timestamps=data.timestamps,
        )
