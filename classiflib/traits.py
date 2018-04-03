"""Custom traits for performing validation."""

import numpy as np
from traits.api import Generic


class OdinEmbeddedWeights(Generic):
    default_value = np.zeros((1, 8))

    def validate(self, obj, name, value):
        if not isinstance(value, np.ndarray):
            self.error(obj, name, value)

        if not value.dtype == np.float:
            self.error(obj, name, value)

        if not len(value.shape) == 2:
            self.error(obj, name, value)

        if not value.shape[1] == 8:
            self.error(obj, name, value)

        return value
