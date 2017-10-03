import time
import numpy as np
from sklearn.base import BaseEstimator

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
    need to be instantiated directly.

    Parameters
    ----------
    classifier : BaseEstimator
        The classifier object
    pairs : np.recarray
        Bipolar pairs used for training (dtype: ``classiflib.dtypes.pairs``)
    powers : np.ndarray
        Mean powers matrix
    frequencies : np.ndarray
        Frequencies the classifier uses.
    weights : np.recarray
        Weights (dtype: ``classiflib.dtypes.weights``). ``None`` if creating a
        container for serialization.
    intercept : float
        Intercept. ``None`` if creating a container for serialization.
    classifier_info : dict
        A dict containing the following keys: classname, subject, roc, auc,
        params
    versions : dict
        All relevant version numbers at serialization time (``None`` if creating
        a container to be serialized).
    timestamp : float
        Unix time in seconds (current time if not given).

    """
    def __init__(self, classifier, pairs, powers, frequencies=FRDefaults,
                 weights=None, intercept=None,
                 classifier_info=_empty_classifier_info, versions=None,
                 timestamp=None):
        self.classifier = classifier
        self.classifier_info = classifier_info
        self.weights = weights
        self.intercept = intercept
        self.mean_powers = powers
        self.frequencies = frequencies
        self.pairs = pairs
        self.versions = versions
        self.timestamp = timestamp or time.time()

    def serialize(self, filename, overwrite=False):
        """Serialize to a file.

        Parameters
        ----------
        filename : str
            Output filename. The serializer used is determined by the extension.
        overwrite : bool
            Wether or not to overwrite an existing file (default: False)

        Notes
        -----
        Currently supported serialization methods:

        * ``.pkl`` -> joblib pickling
        * ``.h5`` -> HDF5

        """
        from .serializers import PickleSerializer, HDF5Serializer

        extension = filename.split('.')[-1]

        try:
            SerializerClass = {
                'pkl': PickleSerializer,
                'h5': HDF5Serializer,
            }[extension]
        except KeyError:
            raise RuntimeError("Unknown file extension: " + extension)

        roc = self.classifier_info.get('roc', None)
        auc = self.classifier_info.get('auc', None)
        subject = self.classifier_info['subject']

        serializer = SerializerClass(
            self.classifier, self.pairs, self.mean_powers, self.frequencies,
            roc=roc, auc=auc, subject=subject, timestamp=self.timestamp
        )

        serializer.serialize(overwrite=overwrite)

    @classmethod
    def load(cls, filename):
        """Load a serialized :class:`ClassifierContainer`.

        Parameters
        ----------
        filename : str

        """
        from .serializers import PickleSerializer, HDF5Serializer

        extension = filename.split('.')[-1]

        try:
            loader = {
                'pkl': PickleSerializer,
                'h5': HDF5Serializer
            }[extension]
        except KeyError:
            raise RuntimeError("Unknown file extension: " + extension)

        return loader.deserialize(filename)
