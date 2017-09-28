from classiflib.dtypes import *


def test_with_id():
    dtype = with_id(pairs)

    assert 'id' in dtype.fields
    for key in pairs.fields:
        assert key in dtype.fields

    array = np.recarray((1,), dtype=dtype)
    assert array.id == 0
