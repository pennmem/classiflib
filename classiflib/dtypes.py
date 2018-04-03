"""Defines reusable dtypes for recarrays and HDF5 tables."""

import numpy as np
import traits.api as traits
from traitschema import Schema

from ._traits import OdinEmbeddedWeights


def with_id(dtype, typestr='<i8', name='id'):
    """Inserts an ID field to the passed dtype.

    Parameters
    ----------
    dtype : np.dtype
        Input dtype
    typestr : str
        Type specifier (default: ``i8``)
    name : str
        Name for the id field.

    Returns
    -------
    dtype with prepended id field.

    """
    descr = [(name, typestr)] + dtype.descr
    return np.dtype(descr)


def make_timing_window(start, end, buffer):
    """Makes a timing window using the appropriate dtype (all times are in
    seconds).

    Parameters
    ----------
    start : float
        Window start time
    end : float
        Window end time
    buffer : float
        Window buffer time

    """
    return np.rec.fromrecords([(start, end, buffer)], dtype=timing_window)[0]


# dtype definitions
###################

# Used for pair info
pairs = np.dtype([
    ('contact0', '<i8'),
    ('contact1', '<i8'),
    ('label0', '|S256'),
    ('label1', '|S256')
])

# Classifier weights
weights = np.dtype([
    ('pair_id', '<i8'),
    ('frequency', '<f8'),
    ('value', '<f8')
])

# Window timings
timing_window = np.dtype([
    ('start_time', '<f8'),
    ('end_time', '<f8'),
    ('buffer', '<f8')
])


class OdinEmbeddedMeta(Schema):
    """OdinEmbeddedMeta info that can be stored in a schema bundle."""
    #: Subject code
    subject = traits.CBytes(desc='subject code', maxlen=16)

    #: Time of creation
    timestamp = traits.CFloat(desc='unix timestamp')

    #: Number of embedded channels
    num_channels = traits.CInt(desc='number of channels')

    #: Number of classifiers
    num_classifiers = traits.CInt(desc='number of classifiers')


class OdinEmbeddedClassifier(Schema):
    """General classifier settings for Odin embedded mode."""
    #: Subject code
    subject = traits.CBytes(desc='subject code', maxlen=16)

    #: Averaging interval in ms
    averaging_interval = traits.Int(desc='averaging interval in ms')

    #: Refractory period in ms
    refractory_period = traits.Int(desc='refractory period in ms')

    #: Stim threshold
    threshold = traits.Int(desc='stim threshold')

    #: Stim duration in ms
    stim_duration = traits.Int(desc='stim duration in ms')

    #: Waveform name (should be of the form ``<stim_channel_name>_wfm``)
    waveform_name = traits.CBytes(desc='associated waveform name', maxlen=32)

    #: Stim channel name
    stim_channel_name = traits.CBytes(desc='associated stim channel name', maxlen=32)

    #: Weights per channel per frequency (shape: ``channels x 8``)
    weights = OdinEmbeddedWeights(desc='weights per channel per frequency')

    def __eq__(self, other):
        return self.subject == other.subject \
            and self.averaging_interval == other.averaging_interval \
            and self.refractory_period == other.refractory_period \
            and self.threshold == other.threshold \
            and self.stim_duration == other.stim_duration \
            and self.waveform_name == other.waveform_name \
            and self.stim_channel_name == other.stim_channel_name \
            and (self.weights == other.weights).all()


class OdinEmbeddedChannel(Schema):
    """Odin embedded mode channel specifications."""
    #: Subject code
    subject = traits.CBytes(desc='subject code', maxlen=16)

    #: Sense channel label
    label = traits.CBytes(desc="sense channel label", maxlen=32)

    #: Mean values per frequency
    means = traits.Array(dtype=np.int16, shape=(8,),
                         desc='raw means values per frequency')

    #: Standard deviations per frequency
    sigmas = traits.Array(dtype=np.int16, shape=(8,),
                          desc='raw standard deviations per frequency')
