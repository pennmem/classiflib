import pytest

from numpy.testing import assert_equal
from traits.api import HasTraits
from traits.trait_errors import TraitError

from classiflib.traits import *


class Thing(HasTraits):
            weights = OdinEmbeddedWeights(desc='test')


@pytest.mark.traits
class TestOdinEmbeddedWeights:
    def test_validation(self):
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

    def test_eq(self):
        thing1 = Thing()
        thing2 = Thing()

        thing1.weights = np.random.random((1, 8))

        with pytest.raises(AssertionError):
            assert_equal(thing1.weights, thing2.weights)

        thing2.weights = thing1.weights
        assert_equal(thing1.weights, thing2.weights)
