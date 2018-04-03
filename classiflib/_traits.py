"""Custom traits for performing validation."""

import numpy as np
from numpy.testing import assert_equal

from traits.api import Generic


class OdinEmbeddedWeights(Generic):
    default_value = np.zeros((1, 8))

    def validate(self, obj, name, value):
        if isinstance(value, np.ndarray):
            if value.dtype == np.float:
                if len(value.shape) == 2:
                    if value.shape[1] == 8:
                        return value
        self.error(obj, name, value)
