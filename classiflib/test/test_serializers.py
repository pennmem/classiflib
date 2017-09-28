import numpy as np
import tables
import pytest

from classiflib import dtypes
from classiflib.serializers import BaseSerializer, HDF5Serializer


@pytest.fixture
def single_pair():
    yield [(0, 1, 'A0', 'A1')]


class DummyClassifier(object):
    pass


class TestBaseSerializer:
    def setup_method(self, method):
        self.serializer = BaseSerializer(DummyClassifier(), [(0, 1, 'A0', 'A1')])

    def test_version(self):
        with pytest.raises(NotImplementedError):
            _ = self.serializer.version


class TestHDF5Serializer:
    @pytest.mark.parametrize('dtype', ['list', 'recarray'])
    def test_serialize(self, dtype, single_pair):
        filename = 'out.h5'
        classifier = DummyClassifier()

        if dtype == 'list':
            pairs = single_pair
        else:
            pairs = np.rec.fromrecords(single_pair, dtype=dtypes.pairs)

        s = HDF5Serializer(classifier, pairs)
        s.serialize(filename)

        with tables.open_file(filename) as hfile:
            group = hfile.root.classifier.info
            assert group.classname.read() == 'classiflib.test.test_serializers.DummyClassifier'.encode()
