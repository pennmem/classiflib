import numpy as np
from sklearn.base import BaseEstimator


class ClassifierContainer(object):
    """Container carrying a classifier and associated data. This is used as the
    serializer-neutral object that is returned when deserializing and should not
    need to be instantiated directly.

    Parameters
    ----------
    classifier : BaseEstimator
        The classifier object
    classifier_info : dict
    weights : np.recarray
        Weights
    intercept : float
        Intercept
    mean_powers : np.recarray (?)
        Mean powers
    pairs : np.recarray
        Bipolar pairs used for training (dtype: ``.dtypes.pairs``)
    versions : dict
        All relevant version numbers at serialization time.

    """
    def __init__(self, classifier, classifier_info, weights, intercept,
                 mean_powers, pairs, versions):
        self.classifier = classifier
        self.classifier_info = classifier_info
        self.weights = weights
        self.intercept = intercept
        self.mean_powers = mean_powers
        self.pairs = pairs
        self.versions = versions
