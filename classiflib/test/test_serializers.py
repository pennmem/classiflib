from contextlib import contextmanager
from io import BytesIO

import pytest
import numpy as np
import tables
from sklearn import __version__ as sklearn_version

from classiflib import dtypes, __version__
from classiflib.classifier import CLASSIFIER_VERSION
from classiflib.serializers import BaseSerializer, HDF5Serializer


@pytest.fixture
def single_pair():
    return [(0, 1, 'A0', 'A1')]


class DummyClassifier(object):
    pass


class TestBaseSerializer:
    def setup_method(self, method):
        self.serializer = BaseSerializer(DummyClassifier(), [(0, 1, 'A0', 'A1')])

    def test_version(self):
        with pytest.raises(NotImplementedError):
            _ = self.serializer.version

    def test_serialize(self):
        with pytest.raises(NotImplementedError):
            self.serializer.serialize("dummy.txt")


class TestHDF5Serializer:
    def setup_method(self, method):
        self.classifier = DummyClassifier()
        self.pairs = single_pair()
        self.serializer = HDF5Serializer(self.classifier, self.pairs, subject='guido')

    @contextmanager
    def hfile(self):
        with tables.open_file('out.h5', 'w') as hfile:
            yield hfile

    @contextmanager
    def hopen(self):
        with tables.open_file('out.h5', 'r') as hfile:
            yield hfile

    def test_add_attributes(self):
        with self.hfile() as hfile:
            self.serializer.add_attributes(hfile)

        with self.hopen() as hfile:
            assert len(hfile.get_node_attr('/', 'commit_hash')) > 1

    def test_add_versions(self):
        with self.hfile() as hfile:
            self.serializer.add_versions(hfile)

        with self.hopen() as hfile:
            vgroup = hfile.root.versions
            assert vgroup.sklearn.read() == sklearn_version.encode()
            assert vgroup.classiflib.read() == __version__.encode()
            assert vgroup.classifier.read() == CLASSIFIER_VERSION.encode()
            assert vgroup.serialization.read() == ("HDF5Serializer_" + HDF5Serializer._version).encode()

    def test_add_pairs(self):
        pairs = [
            (n, n + 1, 'A{}'.format(n), 'A{}'.format(n + 1))
            for n in range(10)
        ]
        serializer = HDF5Serializer(self.classifier, pairs)

        with self.hfile() as hfile:
            serializer.add_pairs(hfile)

        with self.hopen() as hfile:
            for i, row in enumerate(hfile.get_node('/', 'pairs')):
                assert row['id'] == i
                assert row['contact1'] == i
                assert row['contact2'] == i + 1
                assert row['label1'] == 'A{}'.format(i).encode()
                assert row['label2'] == 'A{}'.format(i + 1).encode()

    @pytest.mark.parametrize('dtype', ['list', 'recarray'])
    @pytest.mark.parametrize('overwrite', [True, False])
    def test_serialize(self, dtype, overwrite, single_pair):
        if dtype == 'list':
            pairs = single_pair
        else:
            pairs = np.rec.fromrecords(single_pair, dtype=dtypes.pairs)

        s = HDF5Serializer(self.classifier, pairs)

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
