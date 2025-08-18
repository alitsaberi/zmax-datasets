import numpy as np
from scipy.signal import resample_poly

from zmax_datasets.transforms.base import Transform
from zmax_datasets.utils.data import Data


class Resample(Transform):
    def __call__(
        self,
        data: Data,
        new_sample_rate: int,
        **kwargs,
    ) -> Data:
        return Data(
            resample_poly(
                data.array.astype(np.float64),
                new_sample_rate,
                data.sample_rate,
                axis=0,
                **kwargs,
            ),
            sample_rate=new_sample_rate,
            channel_names=data.channel_names,
            timestamp_offset=data.timestamps[0],
        )
