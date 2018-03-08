"""Defines reusable dtypes for recarrays and HDF5 tables."""

import numpy as np
import traits.api as traits
from traitschema import Schema


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


class Meta(Schema):
    """Meta info that can be stored in a schema bundle."""
    subject = traits.Bytes(desc='subject code', maxlen=16)
    timestamp = traits.Float(desc='unix timestamp')


class OdinEmbeddedClassifier(Schema):
    """General classifier settings for Odin embedded mode."""
    subject = traits.Bytes(desc='subject code', maxlen=16)
    averaging_interval = traits.Int(desc='averaging interval in ms')
    refractory_period = traits.Int(desc='refractory period in ms')
    threshold = traits.Int(desc='stim threshold in dB')
    stim_duration = traits.Int(desc='stim duration in ms')
    waveform_name = traits.Bytes(desc='associated waveform name', maxlen=32)
    stim_channel_name = traits.Bytes(desc='associated stim channel name', maxlen=32)


class OdinEmbeddedChannel(Schema):
    """Odin embedded mode channel specifications."""
    subject = traits.Bytes(desc='subject code', maxlen=16)
    label = traits.Bytes(desc="sense channel label", maxlen=32)
    means = traits.Array(dtype=np.int16, shape=(8,),
                         desc='raw means values per frequency')
    sigmas = traits.Array(dtype=np.int16, shape=(8,),
                          desc='raw standard deviations per frequency')
    weights = traits.Array(dtype=np.float64, shape=(8,),
                           desc='weights per frequency')
