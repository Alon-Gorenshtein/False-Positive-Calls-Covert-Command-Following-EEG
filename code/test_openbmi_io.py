import os
import numpy as np
import pytest
import openbmi_io as ob

MAT = "../output/raw/openbmi/sess01_subj01_EEG_SSVEP.mat"


@pytest.mark.skipif(not os.path.exists(MAT), reason="OpenBMI SSVEP s1 not downloaded")
def test_ssvep_training_run_has_25_trials_per_frequency_and_cca_finds_most():
    x, fs, t, y, chan = ob.load(MAT, "SSVEP", "train")
    S, att = ob.ssvep_scores(x, fs, t, y, chan)
    assert np.bincount(att).tolist() == [25, 25, 25, 25] and (S.argmax(1) == att).mean() > 0.5
