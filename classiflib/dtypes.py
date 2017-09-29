"""Defines reusable dtypes for recarrays and HDF5 tables."""

import numpy as np

# Used for pair info
pairs = np.dtype([
    ('contact1', '<i8'),
    ('contact2', '<i8'),
    ('label1', '|S256'),
    ('label2', '|S256')
])


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
