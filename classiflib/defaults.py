import numpy as np
from .dtypes import make_timing_window


class FRDefaults:  # pragma: nocover
    """Defaults used for the FR classifier."""
    # Timing
    encoding_window = make_timing_window(0., 1.366, 1.365)
    retrieval_window = make_timing_window(-0.525, 0.0, 0.524)
    hfs_window = make_timing_window(0., 1.6, 1.)

    # Butterworth filter parameters
    filter_width = 5
    filter_order = 4

    # Frequencies
    freqs = np.logspace(np.log10(6), np.log10(180), 8)
    hfs = (lambda hfs: hfs[hfs >= 70.0])(np.logspace(np.log10(2), np.log10(200), 50))

    # Encoding samples weight when fitting classifier
    encoding_samples_weight = 2.5
