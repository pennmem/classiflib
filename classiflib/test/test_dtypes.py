from classiflib.dtypes import *


def test_with_id():
    dtype = with_id(pairs)

    assert 'id' in dtype.fields
    for key in pairs.fields:
        assert key in dtype.fields

    array = np.rec.fromrecords([(0, 0, 1, 'A', 'B')], dtype=dtype)
    assert array.id[0] == 0


def test_make_timing_window():
    window = make_timing_window(0., 1.366, 1.365)
    assert window.start_time == 0
    assert window.end_time == 1.366
    assert window.buffer == 1.365
