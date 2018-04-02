import time
import os
import os.path as osp

import numpy as np
from numpy.testing import assert_equal, assert_almost_equal
from sklearn.base import BaseEstimator
from traitschema import bundle_schema, load_bundle

from . import dtypes
from .defaults import FRDefaults

_empty_classifier_info = {
    'classname': None,
    'subject': 'undefined',
    'roc': None,
    'auc': None,
    'params': {}
}


class ClassifierContainer(object):
    """Container carrying a classifier and associated data. This is used as the
    serializer-neutral object that is returned when deserializing and should not
    need to be instantiated directly. All parameters become attributes of the
    same name.

    Parameters
    ----------
    classifier : BaseEstimator
        The classifier object
    pairs : np.recarray
        Bipolar pairs used for training (dtype: ``classiflib.dtypes.pairs``)
    features : np.ndarray
        Features matrix
    events : np.recarray
        Events associated with the features
    sample_weight : np.ndarray
        Sample weights used during training
    frequencies : np.ndarray
        Frequencies the classifier uses.
    weights : np.recarray
        Weights (dtype: ``classiflib.dtypes.weights``). ``None`` if creating a
        container for serialization.
    intercept : float
        Intercept. ``None`` if creating a container for serialization.
    classifier_info : dict
        A dict possibly containing the following keys: classname, subject, roc,
        auc, params (see the meaning of these in the base serializer class).
    versions : dict
        All relevant version numbers at serialization time (``None`` if creating
        a container to be serialized).
    timestamp : float
        Unix time in seconds (current time if not given).

    """
    def __init__(self, classifier, pairs, features, events=None, sample_weight=None,
                 frequencies=FRDefaults.freqs, weights=None, intercept=None,
                 classifier_info=_empty_classifier_info, versions=None,
                 timestamp=None):
        self.classifier = classifier

        if not isinstance(pairs, np.recarray):
            dtype = dtypes.with_id(dtypes.pairs)
            self.pairs = np.rec.fromrecords(pairs, dtype=dtype)
        else:
            self.pairs = pairs

        self.features = features
        self.events = events
        self.sample_weight = sample_weight

        self.frequencies = frequencies
        self.weights = weights
        self.intercept = intercept
        self.classifier_info = classifier_info
        self.versions = versions
        self.timestamp = timestamp or time.time()

    def __eq__(self, other):
        """Check that containers are approximately equal. This uses numpy's
        :func:`assert_almost_equal` function with the default tolerance where
        applicable.

        Parameters
        ----------
        other : ClassifierContainer

        """
        try:
            assert_almost_equal(self.features, other.features)
            assert_equal(self.events, other.events)

            if self.sample_weight is None:
                assert other.sample_weight is None
            else:
                assert_almost_equal(self.sample_weight, other.sample_weight)

            assert_equal(self.frequencies, other.frequencies)

            if self.weights is None:
                assert other.weights is None
            else:
                assert_almost_equal(self.weights, other.weights)

            if self.intercept is None:
                assert other.intercept is None
            else:
                assert_almost_equal(self.intercept, other.intercept)

            assert self.classifier_info['classname'] == other.classifier_info['classname']
            assert self.classifier_info['params'] == other.classifier_info['params']
        except AssertionError:
            return False
        return True

    def save(self, filename, overwrite=False, create_directories=True):
        """Serialize to a file.

        Parameters
        ----------
        filename : str
            Output filename. The serializer used is determined by the extension.
        overwrite : bool
            Whether or not to overwrite an existing file (default: False)
        create_directories : bool
            Recursively create directories for the file if they don't already
            exist.

        Notes
        -----
        Currently supported serialization methods:

        * ``.pkl`` -> joblib pickling
        * ``.h5`` -> HDF5
        * ``.zip`` -> zipped file (similar in structure to HDF5 format)

        """
        from ._serializers import ext_to_class

        extension = filename.split('.')[-1]
        dirs = osp.dirname(filename)

        if len(dirs) and create_directories:
            try:
                os.makedirs(dirs)
            except OSError:
                pass

        try:
            SerializerClass = ext_to_class[extension]
        except KeyError:
            raise RuntimeError("Unknown file extension: " + extension)

        roc = self.classifier_info.get('roc', None)
        auc = self.classifier_info.get('auc', None)
        subject = self.classifier_info['subject']

        serializer = SerializerClass(
            self.classifier, self.pairs, self.features, self.frequencies,
            events=self.events, sample_weight=self.sample_weight,
            roc=roc, auc=auc, subject=subject, timestamp=self.timestamp
        )

        serializer.serialize(filename, overwrite=overwrite)

    @classmethod
    def load(cls, filename):
        """Load a serialized :class:`ClassifierContainer`.

        Parameters
        ----------
        filename : str

        """
        from ._serializers import ext_to_class

        extension = filename.split('.')[-1]

        try:
            loader = ext_to_class[extension]
        except KeyError:
            raise RuntimeError("Unknown file extension: " + extension)

        return loader.deserialize(filename)


class OdinEmbeddedClassifierContainer(object):
    """Container for Odin ENS embedded mode classifiers.

    Parameters
    ----------
    channels : List[List[dtypes.OdinEmbeddedChannel]]
        Channel specifications. Each entry is a list of channels associated with
        each classifier (or just a list of channels for record-only mode). There
        must be 1-32 channels defined.

    classifiers : List[dtypes.OdinEmbeddedClassifier]
        Classifier specifications. Must have 0 (record-only mode), 1, or 2.

    timestamp : float or None
        Timestamp or None to use the current time.

    Raises
    ------
    IndexError
        If number of channels or classifiers is not within allowed limits.

    """
    def __init__(self, channels, classifiers, timestamp=None):
        # validate lengths
        if not 0 < len(channels) <= 2:
            raise IndexError("you must specify 1-2 sets of channels")
        if not 0 <= len(classifiers) <= 2:
            raise IndexError("you must specify 0-2 classifiers")
        for channelset in channels:
            if not 0 < len(channelset) <= 32:
                raise IndexError("you must specify 1-32 channels per classifier")

        # ensure subjects match in all
        subject = channels[0][0].subject
        for channelset in channels:
            for ch in channelset:
                assert ch.subject == subject
            for cl in classifiers:
                assert cl.subject == subject

        self.channels = channels
        self.classifiers = classifiers

        num_channels = [len(channelset) for channelset in self.channels]

        self.meta = dtypes.OdinEmbeddedMeta(
            subject=subject,
            timestamp=(timestamp or time.time()),
            num_channels=num_channels,
            num_classifiers=len(self.classifiers)
        )

    def __eq__(self, other):
        if not self.classifiers == other.classifiers:
            return False
        if not self.meta.timestamp == other.meta.timestamp:
            return False

        for i, channelset in enumerate(self.channels):
            if not channelset == other.channels[i]:
                return False

        return True

    def save(self, filename, overwrite=False, create_directories=True):
        """Serialize to a file."""
        assert filename.endswith('.zip')

        if osp.exists(filename) and not overwrite:
            raise RuntimeError("{} already exists".format(filename))

        if create_directories:
            try:
                os.makedirs(osp.dirname(filename))
            except OSError:
                pass

        schema = {'meta': self.meta}

        for i in range(max(len(self.classifiers), 1)):
            try:
                schema.update({
                    'classifier{}'.format(i): self.classifiers[i]
                })
            except IndexError:
                pass

            schema.update({
                'classifier{}_ch{}'.format(i, j): channel
                for j, channel in enumerate(self.channels[i])
            })

        bundle_schema(filename, schema)

    @classmethod
    def load(cls, filename):
        """Load a saved classifier."""
        schema = load_bundle(filename)
        num_channels = schema['meta'].num_channels
        num_classifiers = schema['meta'].num_classifiers

        channels = []
        for i in range(max(num_classifiers, 1)):
            channels.append([
                schema['classifier{}_ch{}'.format(i, j)]
                for j in range(num_channels[i])
            ])

        classifiers = [schema['classifier{}'.format(i)] for i in range(num_classifiers)]

        return cls(channels, classifiers, schema['meta'].timestamp)
