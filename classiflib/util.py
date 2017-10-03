from subprocess import Popen, PIPE
import json
import numpy as np
from classiflib import dtypes


def git_revision():
    """Return the git revision hash. If the package is being used from
    elsewhere, return 'unknown'.

    """
    p = Popen('git ls-files'.split(), stdout=PIPE, stderr=PIPE)
    stdout, _ = p.communicate()
    if not len(stdout):  # we're not in a git repository
        return 'unknown'

    p = Popen('git rev-parse HEAD'.split(), stdout=PIPE, stderr=PIPE)
    stdout, _ = p.communicate()
    if len(stdout) > 0:
        return stdout.decode().strip()
    else:
        return 'unknown'


def convert_pairs_json(filename):
    """Convert data from ``pairs.json`` to the minimal recarray format.

    Parameters
    ----------
    filename : str

    Returns
    -------
    np.recarray
        dtype = ``classflib.dtypes.pairs``

    """
    with open(filename, 'r') as f:
        pairs_json = json.loads(f.read())

    pairs = pairs_json[list(pairs_json.keys())[0]]['pairs']
    records = [(entry['channel_1'], entry['channel_2'], pair.split('-')[0], pair.split('-')[1])
               for pair, entry in pairs.items()]
    return np.sort(np.rec.fromrecords(records, dtype=dtypes.pairs), order='contact1')


if __name__ == "__main__":
    print(git_revision())
