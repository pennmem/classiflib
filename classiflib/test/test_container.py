from contextlib import contextmanager
from copy import deepcopy
import os.path as osp
import pytest
import numpy as np
from numpy.testing import assert_equal

from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib

from classiflib import dtypes, FRDefaults
from classiflib.container import ClassifierContainer, OdinEmbeddedClassifierContainer


@contextmanager
def duplicate(c):
    duplicated = deepcopy(c)
    yield duplicated


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


def test_save(classifier, pairs, powers, tmpdir):
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

    # Test making directories
    filename = str(tmpdir.join('some').join('path').join('out.zip'))

    with pytest.raises(IOError):
        container.save(filename, create_directories=False)

    container.save(filename, create_directories=True)
    assert osp.exists(filename)


@pytest.mark.parametrize('sample_weight', [None, np.random.random((100,))])
@pytest.mark.parametrize('weights', [None, np.random.random((100,))])
@pytest.mark.parametrize('intercept', [None, np.random.random((1,))[0]])
def test_compare(classifier, pairs, powers, sample_weight, weights, intercept):
    container = ClassifierContainer(classifier, pairs, powers)

    container.sample_weight = sample_weight
    container.weights = weights
    container.intercept = intercept

    with duplicate(container) as copy:
        assert container == copy

    with duplicate(container) as copy:
        copy.features += 1
        assert container != copy

    with duplicate(container) as copy:
        pass  # FIXME: modify event

    with duplicate(container) as copy:
        copy.frequencies[-1] = 10
        assert container != copy

    with duplicate(container) as copy:
        if weights is None:
            copy.weights = 10
        else:
            copy.weights[0] += 1
        assert container != copy

    with duplicate(container) as copy:
        copy.intercept = 100
        assert container != copy

    with duplicate(container) as copy:
        copy.classifier_info['classname'] = 'Nope'
        assert container != copy

    with duplicate(container) as copy:
        copy.classifier_info['params'] = {'nonsense': 'a'}
        assert container != copy


@pytest.mark.odin_embedded
class TestOdinEmbeddedContainer:
    @staticmethod
    def make_channels(subject, count):
        return [
            dtypes.OdinEmbeddedChannel(
                subject=subject,
                label='chan{}'.format(i).encode('ascii'),
                means=np.zeros(8, dtype=np.int16),
                sigmas=np.zeros(8, dtype=np.int16)
            )
            for i in range(count)
        ]

    @staticmethod
    def make_classifiers(subject, count, num_channels):
        return [
            dtypes.OdinEmbeddedClassifier(
                subject=subject,
                averaging_interval=1000,
                refractory_period=5000,
                threshold=-10,
                stim_duration=500,
                waveform_name='wvfm{}'.format(i).encode('ascii'),
                stim_channel_name='chan{}'.format(i).encode('ascii'),
                weights=np.random.uniform(0, 1, (num_channels, 8))
            )
            for i in range(count)
        ]

    def test_create(self):
        channels = []
        classifiers = []
        subject = b'R0001Q'

        # neither channels nor classifiers specified
        with pytest.raises(IndexError):
            OdinEmbeddedClassifierContainer(channels, classifiers)

        # too many channels
        channels = self.make_channels(subject, 33)
        with pytest.raises(IndexError):
            OdinEmbeddedClassifierContainer(channels, classifiers)

        # too many classifiers
        classifiers = self.make_classifiers(subject, 3, 32)
        with pytest.raises(IndexError):
            OdinEmbeddedClassifierContainer(channels, classifiers)

        # within limits
        channels = self.make_channels(subject, 32)
        classifiers.pop()
        oecc = OdinEmbeddedClassifierContainer(channels, classifiers)

        assert oecc.meta.subject == subject

        for i, ch in enumerate(channels):
            assert ch == oecc.channels[i]

        for i, cl in enumerate(classifiers):
            # we have to test these separately because the custom trait for
            # weight validation makes the usual Schema.__eq__ fail...
            assert cl == oecc.classifiers[i]

    @pytest.mark.parametrize('num_classifiers', [0, 1, 2])
    def test_save_load(self, tmpdir, num_classifiers):
        subject = b'R0001X'

        channels = self.make_channels(subject, 32)
        classifiers = self.make_classifiers(subject, num_classifiers, len(channels))
        filename = str(tmpdir.join('classifier.zip'))
        cc = OdinEmbeddedClassifierContainer(channels, classifiers)
        cc.save(filename)

        other = OdinEmbeddedClassifierContainer.load(filename)
        assert cc == other
