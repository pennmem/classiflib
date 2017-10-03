import os.path as osp

from . import data_dir
from .. import dtypes
from ..util import convert_pairs_json


def test_convert_pairs_json():
    filename = osp.join(data_dir, 'pairs.json')
    ra = convert_pairs_json(filename)
    assert len(ra) == 142
    assert ra.dtype == dtypes.pairs
