import numpy as np, eeg_io

def test_load_subject_shapes_and_labels():
    X, y, ch, sf = eeg_io.load_subject(1)
    assert X.ndim == 3 and X.shape[0] == y.shape[0]
    assert X.shape[1] == 64
    assert set(np.unique(y)) <= {0, 1} and 0 in y and 1 in y
    assert 0.3 < y.mean() < 0.7
    assert abs(sf - 160.0) < 1e-3

def test_exclusions_listed():
    subs = eeg_io.all_subjects()
    assert 88 not in subs and 100 not in subs and len(subs) == 105
