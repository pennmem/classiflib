import time
import os.path as osp
from functools import partial
import json
from importlib import import_module
from zipfile import ZipFile, ZIP_DEFLATED
from io import BytesIO
from numbers import Integral

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
        ``(contact0: int, contact1: int, label0: str, label1: str)``. Also can
        be a recarray with the dtype ``.dtypes.pairs``.
    features : np.ndarray
        A MxN matrix of features during training (M = number of events, N =
        number of features).
    frequencies : array-like
        List of frequencies used by the classifier.
    events : np.recarray
        Event data. If given, the length must be the same as the first dimension
        of the features matrix.
    sample_weight : np.ndarray
        Sample weights used when training the classifier.
    roc : np.ndarray
        ROC curve data
    auc : float
        Computed AUC
    subject : str
        Subject ID
    timestamp : float
        Optional timestamp to indicate creation time (now if not given).

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

    def __init__(self, classifier, pairs, features, frequencies=FRDefaults.freqs,
                 events=None, sample_weight=None, roc=None, auc=None,
                 subject="undefined", timestamp=None):
        self.classifier = self._validate_classifier(classifier)
        self.pairs = self._validate_pairs(pairs)

        assert features.shape[1] == len(pairs) * len(frequencies), \
            "Number of features doesn't match power matrix shape!"
        self.features = features

        if events is not None:
            assert len(events) == len(self.features), \
                "Number of events doesn't match feature matrix"
        self.events = events

        self.sample_weight = sample_weight
        self.roc = roc
        self.auc = auc
        self.subject = subject
        self.frequencies = frequencies
        self.timestamp = timestamp or time.time()

        # We have to omit attributes added after training (ones ending with _)
        self.params = {
            key: value
            for key, value in self.classifier.__dict__.items()
            if not key.endswith('_')
        }

        weights = []
        coefs = self.classifier.coef_.flatten()
        pair_id = 0
        for i in range(len(coefs)):
            f_index = i % len(frequencies)
            f = frequencies[f_index]

            if f_index == 0 and i != 0:
                pair_id += 1

            weights += [(pair_id, f, coefs[i])]
        self.weights = np.rec.fromrecords(weights, dtype=dtypes.weights)

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
        dtype = with_id(dtypes.pairs)

        if isinstance(pairs, np.recarray):
            if pairs.dtype == dtypes.pairs:
                rpairs = np.rec.fromrecords([
                    (i, row[0], row[1], row[2], row[3])
                    for i, row in enumerate(pairs)
                ], dtype=dtype)
                return rpairs
            elif pairs.dtype == with_id(dtypes.pairs):
                return pairs

        for row in pairs:
            assert isinstance(row[0], Integral)
            assert isinstance(row[1], Integral)
            assert isinstance(row[2], str)
            assert isinstance(row[3], str)

        size = len(pairs)
        rpairs = np.recarray((size,), dtype=dtype)
        rpairs.id = np.arange(size, dtype='<i8')

        for i in range(size):
            rpairs.contact0[i] = pairs[i][0]
            rpairs.contact1[i] = pairs[i][1]
            rpairs.label0[i] = pairs[i][2]
            rpairs.label1[i] = pairs[i][3]

        return rpairs

    @property
    def classname(self):
        """Return the full class name of the classifier."""
        return '.'.join([self.classifier.__module__,
                         self.classifier.__class__.__name__])

    @property
    def classifier_info(self):
        """Return the classifier info as a dict."""
        return {
            'classname': self.classname,
            'subject': self.subject,
            'roc': self.roc,
            'auc': self.auc,
            'params': json.dumps(self.params),
        }

    @property
    def version(self):
        """Generate a version string for the current class. Implementations
        are required to have a ``_version`` attribute.

        """
        if not hasattr(self, '_version'):
            raise NotImplementedError("Serializers must contain a _version attribute")
        return "_".join([self.__class__.__name__, self._version])

    @property
    def versions(self):
        return {
            'classifier': CLASSIFIER_VERSION,
            'classiflib': __version__,
            'serialization': self._version,
            'sklearn': sklearn_version
        }

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

    @staticmethod
    def deserialize(infile):
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
            pairs=self.pairs,
            features=self.features,
            events=self.events,
            sample_weight=self.sample_weight,
            frequencies=self.frequencies,
            weights=self.weights,
            intercept=self.classifier.intercept_,
            classifier_info=self.classifier_info,
            versions=self.versions
        )

        joblib.dump(container, outfile)

    @staticmethod
    def deserialize(infile):
        return joblib.load(infile)


class HDF5Serializer(BaseSerializer):
    """Utility class to serialize or deserialize a classifier using HDF5."""
    _version = "1.0.0"
    __compression = {
        'compression': 'gzip',
        'compression_opts': 9,
    }

    @staticmethod
    def _group_to_dict(hfile, groupname):
        """Loads all members of a group into a dict.

        Parameters
        ----------
        hfile : h5py.File
        groupname : str

        Returns
        -------
        d : dict

        Notes
        -----
        This function is *not* recursive.

        When the dtype of an array is a string, it is assumed that it contains
        a single utf8-encoded string.

        h5py ``Empty`` datasets are converted to ``None``.

        """
        group = hfile[groupname]
        d = {}
        for member in group:
            if isinstance(group[member].value, h5py.Empty):
                d[member] = None
            elif group[member].dtype.char == 'S':
                d[member] = group[member][0].decode()
            else:
                d[member] = group[member].value
        return d

    def addstring(self, group, name, value, dtype='|S64'):
        """Base function for adding a string to a group; will be partialed to
        only have to specify the group once.

        """
        string = value if not hasattr(value, 'encode') else value.encode()
        group.create_dataset(name, data=[string], dtype=dtype, **self.__compression)

    def add_attributes(self, hfile):
        """Adds root node attributes:

        * commit_hash - git commit hash when serialized from the classiflib git
          repo

        """
        hfile.attrs['commit_hash'] = git_revision()
        hfile.attrs['timestamp'] = self.timestamp

    def add_versions(self, hfile):
        """Create version number node and add relevant versions."""
        group = hfile.create_group('/versions')
        addstring = partial(self.addstring, group)
        classifier_version = CLASSIFIER_VERSION

        addstring('sklearn', sklearn_version)
        addstring('classiflib', __version__)
        addstring('classifier', classifier_version)
        addstring('serialization', self.version)

    def add_pairs(self, hfile):
        """Create and populate pairs table."""
        hfile.create_dataset('/pairs', data=self.pairs, **self.__compression)

    def add_classifier(self, hfile):
        """Create classifier group and add data."""
        cgroup = hfile.create_group('/classifier')

        cgroup.create_dataset('weights', data=self.weights, **self.__compression)
        cgroup.create_dataset('intercept', data=[self.classifier.intercept_])

        info_group = cgroup.create_group('info')
        addstring = partial(self.addstring, info_group)
        addstring('classname', self.classname)
        addstring('subject', self.subject)

        params = json.dumps(self.params)
        addstring('params', params, dtype='|S{:d}'.format(len(params)))

        if self.roc is None:
            info_group.create_dataset('roc', dtype=h5py.Empty('<f8'))
        else:
            info_group.create_dataset('roc', data=self.roc, **self.__compression)

        if self.auc is None:
            info_group.create_dataset('auc', dtype=h5py.Empty('<f8'))
        else:
            info_group.create_dataset('auc', data=[self.auc], **self.__compression)

    def add_training(self, hfile):
        """Add training data."""
        group = hfile.create_group('/classifier/training')

        if self.events is None:
            group.create_dataset('events', dtype=h5py.Empty('f'))
        else:
            group.create_dataset('events', data=self.events, **self.__compression)

        if self.sample_weight is None:
            group.create_dataset('sample_weight', dtype=h5py.Empty('f'))
        else:
            group.create_dataset('sample_weight', data=self.sample_weight, **self.__compression)

    def add_features(self, hfile):
        """Add features."""
        hfile.create_dataset('/classifier/features', data=self.features, **self.__compression)

    def add_frequencies(self, hfile):
        """Add a frequencies dataset."""
        hfile.create_dataset('/frequencies', data=self.frequencies, **self.__compression)

    def serialize_impl(self, outfile, overwrite=True):
        assert isinstance(outfile, str), "HDF5Serializer only supports writing to actual files"
        with h5py.File(outfile, 'w') as hfile:
            self.add_attributes(hfile)
            self.add_versions(hfile)
            self.add_pairs(hfile)
            self.add_classifier(hfile)
            self.add_training(hfile)
            self.add_features(hfile)
            self.add_frequencies(hfile)

    @staticmethod
    def deserialize(infile):
        with h5py.File(infile, 'r') as hfile:
            full_classname = hfile['/classifier/info/classname'][0].decode().split('.')
            params = json.loads(hfile['/classifier/info/params'][0].decode())
            classname = full_classname[-1]
            module = import_module('.'.join(full_classname[:-1]))
            classifier = getattr(module, classname)(**params)

            classifier_info = HDF5Serializer._group_to_dict(hfile, '/classifier/info')

            return ClassifierContainer(
                classifier=classifier,
                pairs=hfile['/pairs'].value,
                features=hfile['/classifier/features'].value,
                events=hfile['/classifier/training/events'].value,
                sample_weight=hfile['/classifier/training/sample_weight'].value,
                weights=hfile['/classifier/weights'].value,
                intercept=hfile['/classifier/intercept'].value,
                classifier_info=classifier_info,
                versions=HDF5Serializer._group_to_dict(hfile, '/versions'),
                timestamp=hfile.attrs['timestamp']
            )


class ZipSerializer(BaseSerializer):
    """Serialize to a zip file.

    Notes
    -----
    Version 1.1.0 of the zip serialization format saves events by extracting
    string fields and storing separately. This allows for cross-Python-version
    compatibility.

    """
    _version = "1.1.0"

    def _npsave(self, array):
        buf = BytesIO()
        np.save(buf, array, allow_pickle=False)
        return buf.getvalue()

    def _zasave(self, zfile, name, array):
        """Saves a numpy array to the zip archive."""
        zfile.writestr(name + '.npy', self._npsave(array))

    def _zjsave(self, zfile, name, dictionary):
        """Saves a JSON string to the zip archive."""
        zfile.writestr(name + '.json',
                       json.dumps(dictionary, indent=2, sort_keys=True).encode())

    def _zesave(self, zfile, name, array):
        """Saves the events recarray to the zip archive.

        This is treated separately from other arrays because string dtypes
        can't be saved without pickling and thus will break across Python
        versions.

        """
        # Find all string dtypes
        non_string_cols = [
            fname for fname in array.dtype.names
            if array[fname].dtype != np.dtype('O')
        ]
        string_cols = [
            fname for fname in array.dtype.names
            if fname not in non_string_cols
        ]

        # Remove them from events and save
        events = array[non_string_cols]
        zfile.writestr(name + '.npy', self._npsave(array))

        # Write string columns to a JSON file
        output = {
            col: [d.encode() for d in array[col]]
            for col in string_cols
        }
        self._zjsave(zfile, name, output)

    def serialize_impl(self, outfile):
        with ZipFile(outfile, 'w') as zfile:
            asave = partial(self._zasave, zfile)
            jsave = partial(self._zjsave, zfile)
            esave = partial(self._zesave, zfile)

            zfile.writestr('/metadata.json', json.dumps({
                'commit': git_revision(),
                'timestamp': self.timestamp
            }, indent=2, sort_keys=True).encode())

            asave('/pairs', self.pairs)
            jsave('/versions', self.versions)

            jsave('/classifier/info', self.classifier_info)
            asave('/classifier/intercept', self.classifier.intercept_)
            asave('/classifier/mean_powers', self.features)
            asave('/classifier/weights', self.weights)
            jsave('/classifier/params', self.params)

            esave('/classifier/training/events', self.events)
            asave('/classifier/training/sample_weight', self.sample_weight)

    @staticmethod
    def deserialize(infile):
        with ZipFile(infile, 'r', compression=ZIP_DEFLATED) as zfile:
            def jload(name):
                with zfile.open(name + '.json') as f:
                    return json.loads(f.read().decode())

            def aload(name):
                with zfile.open(name + '.npy') as f:
                    buf = BytesIO(f.read())
                    buf.seek(0)
                    data = np.load(buf)
                    if len(data.dtype):
                        return data.view(data.dtype, np.rec.recarray)
                    else:
                        return data

            metadata = jload('/metadata')
            versions = jload('/versions')
            pairs = aload('/pairs')
            classifier_info = jload('/classifier/info')
            intercept = aload('/classifier/intercept')
            powers = aload('/classifier/mean_powers')
            weights = aload('/classifier/weights')

            components = classifier_info['classname'].split('.')
            params = json.loads(classifier_info['params'])
            classname = components[-1]
            module = import_module('.'.join(components[:-1]))
            classifier = getattr(module, classname)(**params)
            setattr(classifier, 'coef_', weights.value)
            classifier.coef_.shape = (1, len(classifier.coef_))
            setattr(classifier, 'intercept_', intercept)

            return ClassifierContainer(
                classifier=classifier,
                events=aload('/classifier/training/events'),
                sample_weight=aload('/classifier/training/sample_weight'),
                weights=weights,
                intercept=intercept,
                classifier_info=classifier_info,
                features=powers,
                pairs=pairs,
                versions=versions,
                timestamp=metadata['timestamp']
            )


# Mapping of file extensions to serializers
ext_to_class = {
    'pkl': PickleSerializer,
    'h5': HDF5Serializer,
    'zip': ZipSerializer,
}
