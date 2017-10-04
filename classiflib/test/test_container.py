import pytest
import numpy as np
from numpy.testing import assert_equal

from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib

from classiflib import dtypes, FRDefaults
from classiflib.container import ClassifierContainer


@pytest.fixture
def classifier():
    classifier = LogisticRegression()
    classifier.fit([[1, 2], [4, 5]],
                   [1, 2])
    return classifier


@pytest.fixture
def pairs():
    pairs = np.rec.fromrecords([
        (0, 1, '0', '1'),
        (1, 2, '1', '2')
    ], dtype=dtypes.pairs)
    return pairs


@pytest.fixture
def powers():
    powers = np.random.random((2, 2*8))
    return powers


def test_save(classifier, pairs, powers):
    """Test the default values for all optional parameters when serializing."""
    container = ClassifierContainer(classifier, pairs, powers)

    # Unknown file extensions should fail
    with pytest.raises(RuntimeError):
        container.save('out.txt')

    container.save('out.pkl', overwrite=True)

    # Overwriting shouldn't happen when set to False
    with pytest.raises(RuntimeError):
        container.save('out.pkl', overwrite=False)

    loaded = joblib.load('out.pkl')

    assert loaded.classifier_info['roc'] is None
    assert loaded.classifier_info['auc'] is None
    assert loaded.classifier_info['subject'] == 'undefined'
    assert_equal(loaded.frequencies, FRDefaults.freqs)
    assert_equal(loaded.weights.value, classifier.coef_.flatten())
    assert loaded.intercept == classifier.intercept_
