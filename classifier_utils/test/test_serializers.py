import tables
from classifier_utils.serializers import HDF5Serializer


class DummyClassifier(object):
    pass


class TestHDF5Serializer:
    def test_serialize(self):
        filename = 'out.h5'

        s = HDF5Serializer(DummyClassifier())
        s.serialize(filename)

        with tables.open_file(filename) as hfile:
            cgroup = hfile.root.classifier_info
            assert cgroup.classname.read() == 'classifier_utils.test.test_serializers.DummyClassifier'.encode()
