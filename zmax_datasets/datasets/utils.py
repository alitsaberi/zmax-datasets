from collections.abc import Callable
from typing import Any

import numpy as np


def mapper(mapping: dict[int, Any]) -> Callable[[np.ndarray, Any], np.ndarray]:
    return np.vectorize(mapping.get)
