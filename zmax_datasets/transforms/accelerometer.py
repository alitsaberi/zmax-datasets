import numpy as np

from zmax_datasets.transforms.base import Transform
from zmax_datasets.utils.data import Data


class ProcessDynamicAccelerometer(Transform):
    CHANNEL_NAMES = ["magnitude", "magnitude_derivative"]

    def __call__(self, data: Data) -> Data:
        if data.n_channels != 3:
            raise ValueError(
                "Accelerometer data must have exactly three channels."
                f" Found {data.n_channels} channels."
            )

        magnitude = np.linalg.norm(data.array, axis=1)
        magnitude_derivative = (
            np.diff(magnitude, prepend=magnitude[0]) * data.sample_rate
        )

        return Data(
            array=np.column_stack([magnitude, magnitude_derivative]),
            sample_rate=data.sample_rate,
            timestamps=data.timestamps,
            channel_names=self.CHANNEL_NAMES,
        )


class ProcessGravityAccelerometer(Transform):
    CHANNEL_NAMES = ["pitch", "roll"]

    def __call__(self, data: Data) -> Data:
        if data.n_channels != 3:
            raise ValueError(
                "Accelerometer data must have exactly three channels."
                f" Found {data.n_channels} channels."
            )

        x = data.array[:, 0]
        y = data.array[:, 1]
        z = data.array[:, 2]

        pitch = np.arctan2(-x, np.sqrt(y**2 + z**2))
        roll = np.arctan2(y, z)

        return Data(
            array=np.column_stack([pitch, roll]),
            sample_rate=data.sample_rate,
            timestamps=data.timestamps,
            channel_names=self.CHANNEL_NAMES,
        )
