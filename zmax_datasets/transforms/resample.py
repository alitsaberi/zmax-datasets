import math

import numpy as np
from scipy.signal import resample_poly

from zmax_datasets.transforms.base import Transform
from zmax_datasets.utils.data import Data


class Resample(Transform):
    def __init__(self, new_sample_rate: float):
        self.new_sample_rate = new_sample_rate

    def __call__(
        self,
        data: Data,
        **kwargs,
    ) -> Data:
        # Calculate resampling factors
        gcd = math.gcd(int(self.new_sample_rate * 1000), int(data.sample_rate * 1000))
        up = int(self.new_sample_rate * 1000) // gcd
        down = int(data.sample_rate * 1000) // gcd

        # Resample using integer factors
        resampled = Data(
            resample_poly(
                data.array.astype(np.float64),
                up,
                down,
                axis=0,
                **kwargs,
            ),
            sample_rate=self.new_sample_rate,
            channel_names=data.channel_names,
            timestamp_offset=data.timestamps[0],
        )

        return resampled
