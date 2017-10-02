import os.path as osp
from functools import partial
import json

import numpy as np
import h5py
from sklearn import __version__ as sklearn_version
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib

from . import __version__, dtypes
from .dtypes import with_id
from .defaults import FRDefaults
from .classifier import CLASSIFIER_VERSION
from .container import ClassifierContainer
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
    frequencies : array-like
        List of frequencies used by the classifier.
    roc : np.ndarray
        ROC curve data
    auc : float
        Computed AUC
    subject : str
        Subject ID

    Notes
    -----
    The ``pairs`` parameter must explicitly exclude pairs that are excluded from
    the classifier.

    Only :class:`LogisticRegression` classifiers are supported at this time.
    This is due to wanting to ensure maximum floating point precision is
    preserved.

    """
    SUPPORTED_CLASSIFIERS = (
        LogisticRegression,
    )

    def __init__(self, classifier, pairs, frequencies=FRDefaults.freqs,
                 roc=None, auc=None, subject="undefined"):
        # Indicates if this was generated from a legacy pickle file or not
        self._from_legacy_format = False

        self.classifier = self._validate_classifier(classifier)
        self.pairs = self._validate_pairs(pairs)
        self.roc = roc if roc is not None else np.zeros((2, 1))
        self.auc = auc or 0.
        self.subject = subject

        # We have to omit attributes we add for testing purposes
        self.params = {
            'key': value
            for key, value in self.classifier.__dict__.items()
            if not key.endswith('_')
        }

        weights = []
        coefs = self.classifier.coefs_.flatten()
        for i in range(len(coefs)):
            f = frequencies[i % len(frequencies)]
            weights += [(i, f, coefs[i])]
        self.weights = np.rec.fromrecords(weights, dtype=dtypes.weights)

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
        """Validate pairs and convert to recarray if not already."""
        if isinstance(pairs, np.recarray):
            assert pairs.dtype == dtypes.pairs
            return pairs

        for row in pairs:
            assert isinstance(row[0], int)
            assert isinstance(row[1], int)
            assert isinstance(row[2], str)
            assert isinstance(row[3], str)

        dtype = with_id(dtypes.pairs)
        size = len(pairs)
        rpairs = np.recarray((size,), dtype=dtype)
        rpairs.id = np.arange(size, dtype='<i8')

        for i in range(size):
            rpairs.contact1[i] = pairs[i][0]
            rpairs.contact2[i] = pairs[i][1]
            rpairs.label1[i] = pairs[i][2]
            rpairs.label2[i] = pairs[i][3]

        return rpairs

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

    def deserialize(self, infile):
        """Deserialize and return the classifier and associated data.

        Parameters
        ----------
        infile : str
            path to serialized classifier

        Returns
        -------
        container : ClassifierContainer

        Notes
        -----
        This method must be implemented and return a :class:`ClassifierContainer`
        object.

        """
        raise NotImplementedError


class PickleSerializer(BaseSerializer):
    """Uses :mod:`sklearn.externals.joblib` to serialize a :class:`ClassifierContainer`
    as a pickle.

    """
    _version = "1.0.0"

    def serialize_impl(self, outfile):
        container = ClassifierContainer(
            classifier=self.classifier,
            classifier_info={
                'classname': self.classname,
                'subject': self.subject,
                'roc': self.roc,
                'auc': self.auc,
                'params': json.dumps(self.params),
            },
            weights=self.weights,
            intercept=None,  # FIXME
            mean_powers=None,  # FIXME
            pairs=self.pairs,
            versions={
                'classifier': CLASSIFIER_VERSION,
                'classiflib': __version__,
                'serialization': self._version,
                'sklearn': sklearn_version
            }
        )

        joblib.dump(container, outfile)

    def deserialize(self, infile):
        return joblib.load(infile)


class HDF5Serializer(BaseSerializer):
    """Utility class to serialize or deserialize a classifier using HDF5."""
    _version = "1.0.0"

    def addstring(self, group, name, value, dtype='|S64'):
        """Base function for adding a string to a group; will be partialed to
        only have to specify the group once.

        """
        string = value if not hasattr(value, 'encode') else value.encode()
        group.create_dataset(name, data=[string], dtype=dtype)

    def add_attributes(self, hfile):
        """Adds root node attributes:

        * commit_hash - git commit hash when serialized from the classiflib git
          repo

        """
        hfile.attrs['commit_hash'] = git_revision()

    def add_versions(self, hfile):
        """Create version number node and add relevant versions."""
        group = hfile.create_group('/versions')
        addstring = partial(self.addstring, group)
        classifier_version = CLASSIFIER_VERSION if not self._from_legacy_format else "1.0.0"

        addstring('sklearn', sklearn_version)
        addstring('classiflib', __version__)
        addstring('classifier', classifier_version)
        addstring('serialization', self.version)

    def add_pairs(self, hfile):
        """Create and populate pairs table."""
        hfile.create_dataset('/pairs', data=self.pairs, chunks=True)

    def add_classifier(self, hfile):
        """Create classifier group and add data."""
        cgroup = hfile.create_group('/classifier')

        cgroup.create_dataset('weights', data=self.weights, chunks=True)

        info_group = cgroup.create_group('info')
        addstring = partial(self.addstring, info_group)
        addstring('classname', self.classname)
        addstring('subject', self.subject)
        params = json.dumps(self.params)
        addstring('params', params, dtype='|S{:d}'.format(len(params)))
        info_group.create_dataset('roc', data=self.roc, chunks=True)
        info_group.create_dataset('auc', data=[self.auc], chunks=True)

    def _create_hdf5(self, filename):
        with h5py.File(filename, 'w') as hfile:
            self.add_attributes(hfile)
            self.add_versions(hfile)
            self.add_pairs(hfile)
            self.add_classifier(hfile)

    def serialize_impl(self, outfile, overwrite=True):
        assert isinstance(outfile, str), "HDF5Serializer only supports writing to actual files"
        self._create_hdf5(outfile)
