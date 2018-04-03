import pytest

from traits.api import HasTraits
from traits.trait_errors import TraitError

from classiflib.traits import *


@pytest.mark.traits
class TestTraits:
    # @pytest.mark.parametrize(
    #     'array',
    #     [
    #         np.linspace(0, 1, 100),
    #         0,
    #         [1, 2, 3],
    #
    #     ]
    # )
    def test_odin_embedded_weights(self):
        class Thing(HasTraits):
            weights = OdinEmbeddedWeights(desc='test')

        thing = Thing()
        weights = [0] * 8

        # scalar
        with pytest.raises(TraitError):
            thing.weights = 0

        # wrong dtype
        with pytest.raises(TraitError):
            thing.weights = np.array([weights], dtype=int)

        # list
        with pytest.raises(TraitError):
            thing.weights = [1, 2, 3]

        # array but wrong shape
        with pytest.raises(TraitError):
            thing.weights = np.array([weights[:-1]], dtype=float)

        thing.weights = np.array([weights], dtype=float)
