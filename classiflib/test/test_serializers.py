import tables
from classiflib.serializers import HDF5Serializer


class DummyClassifier(object):
    pass


class TestHDF5Serializer:
    def test_serialize(self):
        filename = 'out.h5'
        classifier = DummyClassifier()
        pairs = [(0, 1, 'A0', 'A1')]

        s = HDF5Serializer(classifier, pairs)
        s.serialize(filename)

        with tables.open_file(filename) as hfile:
            cgroup = hfile.root.classifier_info
            assert cgroup.classname.read() == 'classiflib.test.test_serializers.DummyClassifier'.encode()
