import os.path as osp
from functools import partial
from warnings import warn

import numpy as np
import tables
from sklearn import __version__ as sklearn_version
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib

from . import __version__, dtypes
from .dtypes import with_id
from .classifier import CLASSIFIER_VERSION
from .util import git_revision


class BaseSerializer(object):
    """Base classifier serializer class.

    Parameters
    ----------
    classifier : sklearn.base.BaseEstimator
        Trained classifier instance
    pairs : list or np.recarray
        A list containing contact numbers and labels of the pairs used by the
        classifier. Each element of the list is a tuple of the following form:
        ``(contact1: int, contact2: int, label1: str, label2: str)``. Also can
        be a recarray with the dtype ``.dtypes.pairs``.

    Notes
    -----
    Only :class:`LogisticRegression` classifiers are supported at this time.
    This is due to wanting to ensure maximum floating point precision is
    preserved.

    """
    SUPPORTED_CLASSIFIERS = (
        LogisticRegression,
    )

    def __init__(self, classifier, pairs, subject="undefined"):
        # Indicates if this was generated from a legacy pickle file or not
        self._from_legacy_format = False

        self.classifier = self._validate_classifier(classifier)
        self.pairs = self._validate_pairs(pairs)
        self.subject = subject

    @classmethod
    def from_pickle(cls, pickle_file, pairs):
        """Convert the legacy pickle format to other formats.

        Parameters
        ----------
        pickle_file : str or pathlib.Path
            Pickle file to read
        pairs : list or np.recarray
            See :meth:`BaseSerializer.__init__` for details

        """
        classifier = joblib.load(pickle_file)
        serializer = cls(classifier, pairs)
        serializer._from_legacy_format = True
        return serializer

    @staticmethod
    def _validate_classifier(classifier):
        allowed = BaseSerializer.SUPPORTED_CLASSIFIERS
        allowed_str = [c.__name__ for c in allowed]
        assert isinstance(classifier, allowed), \
            "Only the following are supported: " + "\n".join(allowed_str)
        return classifier

    @staticmethod
    def _validate_pairs(pairs):
        for row in pairs:
            if isinstance(row, np.record):
                break
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

    @property
    def version(self):
        """Generate a version string for the current class. Implementations
        are required to have a ``_version`` attribute.

        """
        if not hasattr(self, '_version'):
            raise NotImplementedError("Serializers must contain a _version attribute")
        return "_".join([self.__class__.__name__, self._version])

    def serialize_impl(self, outfile):
        """The method which implements the actual serialization. Must be defined
        in child classes.

        Parameters
        ----------
        outfile : str or file-like

        """
        raise NotImplementedError

    def serialize(self, outfile, overwrite=True):
        """Serialize the data to the specified location. Child classes should
        instead implement the :meth:`serialize_impl` method which takes only the
        ``outfile`` parameter.

        Parameters
        ----------
        outfile : str or file-like
        overwrite : bool

        """
        if isinstance(outfile, str):
            if osp.exists(outfile) and not overwrite:
                raise RuntimeError("{} already exists".format(outfile))

        self.serialize_impl(outfile)


class PickleSerializer(BaseSerializer):
    """A thin wrapper around joblib's pickling (legacy format)."""
    _version = "1.0.0"

    def serialize_impl(self, outfile):
        msg = "Pickling could potentially be harmful; consider another format"
        warn(msg, DeprecationWarning)
        joblib.dump(self.classifier, outfile)


class HDF5Serializer(BaseSerializer):
    """Utility class to serialize or deserialize a classifier using HDF5."""
    _version = "1.0.0"

    def addstring(self, hfile, group, name, value):
        """Base function for adding a string to a group; will be partialed to
        only have to specify the group once.

        """
        hfile.create_array(group, name, obj=str.encode(value))

    def add_attributes(self, hfile):
        """Adds root node attributes:

        * commit_hash - git commit hash when serialized from the classiflib git
          repo

        """
        hfile.set_node_attr('/', 'commit_hash', git_revision())

    def add_versions(self, hfile):
        """Create version number node and add relevant versions."""
        group = hfile.create_group('/', 'versions')
        addstring = partial(self.addstring, hfile, group)
        addstring('sklearn', sklearn_version)
        addstring('classiflib', __version__)
        addstring('classifier', CLASSIFIER_VERSION if not self._from_legacy_format else "1.0.0")
        addstring('serialization', self.version)

    def add_pairs(self, hfile):
        """Create and populate pairs table."""
        hfile.create_table('/', 'pairs', with_id(dtypes.pairs),
                           title="Bipolar pairs", expectedrows=256)
        for i, pair in enumerate(self.pairs):
            row = hfile.root.pairs.row
            row['id'] = i
            row['contact1'] = pair[0]
            row['contact2'] = pair[1]
            row['label1'] = pair[2]
            row['label2'] = pair[3]
            row.append()

    def add_classifier(self, hfile):
        """Create classifier group and add data."""
        cgroup = hfile.create_group('/', 'classifier')
        # hfile.create_table(cgroup, 'weights', Weight, title='Classifier weights',
        #                    expectedrows=256)

        info_group = hfile.create_group(cgroup, 'info')
        addstring = partial(self.addstring, hfile, info_group)
        addstring('classname', self.classname)
        addstring('subject', self.subject)

    def _create_hdf5(self, filename):
        with tables.open_file(filename, 'w') as hfile:
            self.add_attributes(hfile)
            self.add_versions(hfile)
            self.add_pairs(hfile)
            self.add_classifier(hfile)

    def serialize_impl(self, outfile, overwrite=True):
        assert isinstance(outfile, str), "HDF5Serializer only supports writing to actual files"
        self._create_hdf5(outfile)
