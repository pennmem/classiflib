import os.path as osp

import numpy as np
import tables
from sklearn import __version__ as sklearn_version

from . import __version__, dtypes
from .dtypes import with_id
from .classifier import CLASSIFIER_VERSION
from .util import git_revision


class BaseSerializer(object):
    """Base classifier serializer class.

    Parameters
    ----------
    classifier : object
        Trained classifier instance
    pairs : list
        A list containing contact numbers and labels of the pairs used by the
        classifier. Each element of the list is a tuple of the following form:
        ``(contact1: int, contact2: int, label1: str, label2: str)``

    """
    def __init__(self, classifier, pairs):
        self.classifier = classifier
        self.pairs = self._validate_pairs(pairs)

    @staticmethod
    def _validate_pairs(pairs):
        for row in pairs:
            assert isinstance(row[0], int)
            assert isinstance(row[1], int)
            assert isinstance(row[2], str)
            assert isinstance(row[3], str)
        return pairs

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
        # Create empty tables
        hfile.create_table('/', 'pairs', with_id(dtypes.pairs),
                           title="Bipolar pairs", expectedrows=256)
        # hfile.create_table('/', 'weights', Weight, title='Classifier weights',
        #                    expectedrows=256)

        # Populate pair info
        for i, pair in enumerate(self.pairs):
            print(pair)
            row = hfile.root.pairs.row
            row['id'] = i
            row['contact1'] = pair[0]
            row['contact2'] = pair[1]
            row['label1'] = pair[2]
            row['label2'] = pair[3]
            row.append()

        # Add metadata
        group = hfile.create_group('/', 'classifier_info')
        addstring = lambda name, value: hfile.create_array(group, name, obj=str.encode(value))
        addstring('classname', self.classname)
        addstring('sklearn_version', sklearn_version)
        addstring('classifier_version', CLASSIFIER_VERSION)
        addstring('classifier_utils_version', __version__)
        addstring('serialization_version', self.__version)
        addstring('commit_hash', git_revision())

    def serialize(self, filename, overwrite=True):
        """Serialize the data to the specified location."""
        if osp.exists(filename) and not overwrite:
            raise RuntimeError("{} already exists".format(filename))

        with tables.open_file(filename, 'w') as hfile:
            self._create_hdf5(hfile)
