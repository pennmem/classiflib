CLASSIFIER_VERSION = "1.0.1"


def load(filename):
    """Load a serialized classifier and its metadata. This function detects the
    serialization format based on file extension:

    * .h5 -> HDF5 serialization
    * .pkl -> pickled serialization

    Parameters
    ----------
    filename : str
        Path to serialized classifier.

    """
