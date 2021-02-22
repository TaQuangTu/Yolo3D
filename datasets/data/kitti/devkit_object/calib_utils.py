import numpy as np


def get_calib_P2(path):
    with open(path) as f:
        K = [line.split()[1:] for line in f.read().splitlines() if line.startswith('P2:')]
        assert len(K) > 0, 'P2 is not included in %s' % path
        return np.array(K[0], dtype=np.float32)