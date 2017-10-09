import time
import numpy as np
from sklearn.base import BaseEstimator

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

    def save(self, filename, overwrite=False):
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
        * ``.zip`` -> zipped file (similar in structure to HDF5 format)

        """
        from ._serializers import ext_to_class

        extension = filename.split('.')[-1]

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
