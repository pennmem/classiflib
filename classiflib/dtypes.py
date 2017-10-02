"""Defines reusable dtypes for recarrays and HDF5 tables."""

import numpy as np


def with_id(dtype, typestr='i8', name='id'):
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
    types = [(name, typestr)]
    return np.dtype(types + [(key, value[0].str) for key, value in dtype.fields.items()])


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
    ('contact1', '<i8'),
    ('contact2', '<i8'),
    ('label1', '|S256'),
    ('label2', '|S256')
])

# Logistic regression parameters used for instantiation
logit_params = np.dtype([
    ('C', '<f8'),
])

# Window timings
timing_window = np.dtype([
    ('start_time', '<f8'),
    ('end_time', '<f8'),
    ('buffer', '<f8')
])
