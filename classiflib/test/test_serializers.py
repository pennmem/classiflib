import numpy as np
import tables
import pytest

from classiflib import dtypes
from classiflib.serializers import HDF5Serializer


class DummyClassifier(object):
    pass


class TestHDF5Serializer:
    @pytest.mark.parametrize('dtype', ['list', 'recarray'])
    def test_serialize(self, dtype):
        filename = 'out.h5'
        classifier = DummyClassifier()

        data = [(0, 1, 'A0', 'A1')]
        if dtype == 'list':
            pairs = data
        else:
            pairs = np.rec.fromrecords(data, dtype=dtypes.pairs)

        s = HDF5Serializer(classifier, pairs)
        s.serialize(filename)

        with tables.open_file(filename) as hfile:
            cgroup = hfile.root.classifier_info
            assert cgroup.classname.read() == 'classiflib.test.test_serializers.DummyClassifier'.encode()
