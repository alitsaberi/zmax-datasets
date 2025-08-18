from typing import Protocol, runtime_checkable

from zmax_datasets.utils.data import Data


@runtime_checkable
class Transform(Protocol):
    def __call__(self, data: Data, **kwargs) -> Data:
        """
        Protocol for transform functions that process Data objects.

        Args:
            data (Data): Input data object containing array and sample rate
            **kwargs: Additional arguments specific to each transform implementation

        Returns:
            Data: Transformed data object
        """
        ...
