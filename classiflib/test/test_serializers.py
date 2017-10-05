from contextlib import contextmanager
from io import BytesIO
import json

import pytest
import numpy as np
from numpy.testing import assert_equal
import h5py

from sklearn import __version__ as sklearn_version
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.externals import joblib

from classiflib import dtypes, __version__
from classiflib.classifier import CLASSIFIER_VERSION
from classiflib._serializers import BaseSerializer, PickleSerializer, HDF5Serializer, ZipSerializer


@pytest.fixture
def single_pair():
    return [(0, 1, 'A0', 'A1')]


@pytest.fixture
def mean_powers():
    return np.random.random((1000, 8))


class DummyClassifier(LogisticRegression):
    coef_ = np.random.random((1, 8 * 1))
    intercept_ = 0.


class TestBaseSerializer:
    def setup_method(self, method):
        self.serializer = BaseSerializer(DummyClassifier(), [(0, 1, 'A0', 'A1')], mean_powers())

    def test_classifier_validation(self, mean_powers):
        class BadClassifier(object):
            pass

        with pytest.raises(AssertionError):
            BaseSerializer(BadClassifier(), single_pair(), mean_powers)

        with pytest.raises(AssertionError):
            BaseSerializer(SGDClassifier, single_pair(), mean_powers)

    def test_weights(self, mean_powers):
        pairs = single_pair()
        serializer = BaseSerializer(DummyClassifier(), pairs, mean_powers)
        weights = serializer.weights
        assert hasattr(weights, 'pair_id')
        assert hasattr(weights, 'frequency')
        assert hasattr(weights, 'value')
        assert (weights.value == DummyClassifier.coef_[0, :]).all()

    def test_version(self):
        with pytest.raises(NotImplementedError):
            _ = self.serializer.version

    def test_serialize(self):
        with pytest.raises(NotImplementedError):
            self.serializer.serialize("dummy.txt")


class TestPickleSerializer:
    def test_serialize(self, tmpdir, mean_powers):
        serializer = PickleSerializer(DummyClassifier(), single_pair(), mean_powers)

        outfile = tmpdir.join('out.pkl').strpath
        serializer.serialize(outfile)

        container = joblib.load(outfile)
        classifier = container.classifier
        assert isinstance(classifier, DummyClassifier)

        assert container.intercept == 0.

        weights = container.weights
        assert len(weights) == 8

        info = container.classifier_info
        assert 'classname' in info
        assert 'subject' in info
        assert 'roc' in info
        assert 'auc' in info
        assert isinstance(json.loads(info['params']), dict)

    @pytest.mark.skip(reason="not implemented yet")
    def test_deserialize(self):
        pass


class TestHDF5Serializer:
    def setup_method(self, method):
        self.classifier = DummyClassifier()
        self.pairs = single_pair()
        self.mean_powers = mean_powers()
        self.serializer = HDF5Serializer(self.classifier, self.pairs,
                                         self.mean_powers, subject='guido')

    @contextmanager
    def hfile(self):
        with h5py.File('out.h5', 'w') as hfile:
            yield hfile

    @contextmanager
    def hopen(self):
        with h5py.File('out.h5', 'r') as hfile:
            yield hfile

    def test_group_to_dict(self):
        scalar = 1
        matrix = np.random.random((100, 100))

        with self.hfile() as hfile:
            hfile['/group/a'] = scalar
            hfile['/group/b'] = matrix
            hfile['/group/string'] = ['string'.encode()]

        with self.hopen() as hfile:
            d = self.serializer._group_to_dict(hfile, 'group')
            assert d['a'] == scalar
            assert_equal(d['b'], matrix)
            assert d['string'] == 'string'

    def test_add_attributes(self):
        with self.hfile() as hfile:
            self.serializer.add_attributes(hfile)

        with self.hopen() as hfile:
            assert len(hfile.attrs['commit_hash']) > 1

    def test_add_versions(self):
        with self.hfile() as hfile:
            self.serializer.add_versions(hfile)

        with self.hopen() as hfile:
            vgroup = hfile['/versions']
            assert vgroup['sklearn'][0] == sklearn_version.encode()
            assert vgroup['classiflib'][0] == __version__.encode()
            assert vgroup['classifier'][0] == CLASSIFIER_VERSION.encode()
            assert vgroup['serialization'][0] == ("HDF5Serializer_" + HDF5Serializer._version).encode()

    def test_add_pairs(self):
        pairs = [
            (n, n + 1, 'A{}'.format(n), 'A{}'.format(n + 1))
            for n in range(10)
        ]
        powers = np.random.random((1000, 8*len(pairs)))
        serializer = HDF5Serializer(self.classifier, pairs, powers)

        with self.hfile() as hfile:
            serializer.add_pairs(hfile)

        with self.hopen() as hfile:
            for i, row in enumerate(hfile['/pairs']):
                assert row['id'] == i
                assert row['contact1'] == i
                assert row['contact2'] == i + 1
                assert row['label1'] == 'A{}'.format(i).encode()
                assert row['label2'] == 'A{}'.format(i + 1).encode()

    def test_add_powers(self):
        with self.hfile() as hfile:
            self.serializer.add_powers(hfile)

        with self.hopen() as hfile:
            assert_equal(hfile['/classifier/mean_powers'][:], self.mean_powers)

    def test_add_classifier(self):
        roc = np.array([np.linspace(0, 1, 100), np.linspace(0, 1, 100)])
        classifier = DummyClassifier()
        classifier.coef_ = np.random.random((1, 16))
        powers = np.random.random((1000, 2*8))
        serializer = HDF5Serializer(classifier, single_pair() + single_pair(),
                                    powers, roc=roc, auc=0.5)

        with self.hfile() as hfile:
            serializer.add_classifier(hfile)

        with self.hopen() as hfile:
            igroup = hfile['/classifier/info']
            assert igroup['roc'].shape == (2, 100)
            assert igroup['auc'][0] == 0.5
            assert 'classname' in igroup
            assert 'subject' in igroup
            assert isinstance(json.loads(hfile['/classifier/info/params'][0].decode()), dict)

            weights = hfile['/classifier/weights']
            assert weights.shape == (16,)

            intercept = hfile['/classifier/intercept'][0]
            assert intercept == 0.

    @pytest.mark.parametrize('dtype', ['list', 'recarray'])
    @pytest.mark.parametrize('overwrite', [True, False])
    def test_serialize(self, dtype, overwrite, single_pair):
        if dtype == 'list':
            pairs = single_pair
        else:
            pairs = np.rec.fromrecords(single_pair, dtype=dtypes.pairs)

        s = HDF5Serializer(self.classifier, pairs, self.mean_powers)

        if not overwrite:
            with open('out.h5', 'w') as f:
                f.write('')

        try:
            s.serialize('out.h5')
        except RuntimeError as e:
            if not overwrite:
                assert "already exists" in str(e)
            else:
                raise

    def test_serialize_to_buffer(self):
        with pytest.raises(AssertionError):
            self.serializer.serialize(BytesIO())

    def test_deserialize(self):
        self.serializer.serialize('out.h5')

        container = self.serializer.deserialize('out.h5')

        assert isinstance(container.classifier, DummyClassifier)
        assert container.timestamp == self.serializer.timestamp

        info = container.classifier_info
        assert info['auc'] is None
        assert info['roc'] is None
        assert info['subject'] == 'guido'
        assert info['classname'].endswith('DummyClassifier')

        assert_equal(container.weights['value'], self.classifier.coef_[0])
        assert container.intercept == self.classifier.intercept_

        pairs = np.rec.fromrecords(self.pairs, dtype=dtypes.pairs)
        assert_equal(container.pairs.contact1, pairs.contact1)
        assert_equal(container.pairs.contact2, pairs.contact2)
        assert_equal(container.pairs.label1, pairs.label1)
        assert_equal(container.pairs.label2, pairs.label2)

        assert_equal(container.powers, self.mean_powers)
        assert_equal(container.frequencies, self.serializer.frequencies)


@pytest.mark.only
def test_zip_serializer():
    import os

    try:
        os.remove('out.zip')
    except:
        pass

    classifier = DummyClassifier()
    serializer = ZipSerializer(classifier, single_pair(), mean_powers())
    serializer.serialize('out.zip', overwrite=True)

    container = ZipSerializer.deserialize('out.zip')
    assert isinstance(container.classifier, DummyClassifier)
