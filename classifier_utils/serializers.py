import os.path as osp

import tables
from tables import IsDescription, Int32Col, StringCol, StringAtom
from sklearn import __version__ as sklearn_version

from .classifier import CLASSIFIER_VERSION


class Pair(IsDescription):
    """Schema for a pair used by a classifier."""
    id = Int32Col()
    contact1 = Int32Col()
    contact2 = Int32Col()
    label1 = StringCol(itemsize=256)
    label2 = StringCol(itemsize=256)


class Weight(IsDescription):
    """Classifier weights."""
    pair_id = Int32Col()


class BaseSerializer(object):
    """Base classifier serializer class."""
    def __init__(self, classifier):
        self.classifier = classifier

    @property
    def classname(self):
        """Return the full class name of the classifier."""
        return '.'.join([self.classifier.__module__,
                         self.classifier.__class__.__name__])

    def serialize(self, *args, **kwargs):
        raise NotImplementedError


class HDF5Serializer(BaseSerializer):
    """Utility class to serialize or deserialize a classifier using HDF5."""
    __version = "1.0.0"

    def _create_hdf5(self, hfile):
        hfile.create_table('/', 'pairs', Pair, title="Bipolar pairs",
                           expectedrows=256)
        hfile.create_table('/', 'weights', Weight, title='Classifier weights',
                           expectedrows=256)

        group = hfile.create_group('/', 'classifier_info')
        hfile.create_array(group, 'classname', obj=self.classname.encode())
        hfile.create_array(group, 'sklearn_version', obj=sklearn_version.encode())
        hfile.create_array(group, 'classifier_version', obj=CLASSIFIER_VERSION.encode())

        group = hfile.create_group('/', 'serialization')
        hfile.create_array(group, 'serialization_version', obj=self.__version.encode())
        # hfile.create_array(group, 'commit_hash')

    def serialize(self, filename, overwrite=True):
        """Serialize the data to the specified location."""
        if osp.exists(filename) and not overwrite:
            raise RuntimeError("{} already exists".format(filename))

        with tables.open_file(filename, 'w') as hfile:
            self._create_hdf5(hfile)
