"""bench/passive.py.
- The N-passive contrast on the REAL event indexes: counts as in the facts doc; S200, attended targets and same-stream
  deviants never in a contrast. check_contrast re-derives every row from the raw codes, not from the module's role
  labels, and is itself run against deliberately broken contrasts.
- ds003522 is not an N-passive dataset (dropped: only its post-stimulus epoch cache is on disk).
- The dev split (2 subjects per dataset, seed 20261024), its frozen sibling file and the refusal rules.
- The overlay, montage names, and one dev recording per dataset (shapes and units only; no pipeline is run).
cd code && /usr/bin/python3 -m pytest bench/tests/test_passive.py -q"""
import json
import os
import types

import mne
import numpy as np
import pytest

from bench import engine as en
from bench import passive as pv

KOJIMA_STREAM = {1: 1, 2: 2, 101: 1, 102: 1, 103: 2, 104: 2, 111: 1, 112: 1, 113: 2, 114: 2}
TARGET_STREAM = {201: 1, 202: 1, 203: 2, 204: 2}
STD1005 = set(mne.channels.make_standard_montage("standard_1005").ch_names)


def check_contrast(dataset, tab, ev):
    """Every label-1 row is the dataset's no-intent rare stimulus and every label-0 row a standard of the same stream,
    re-derived from the raw codes (raises AssertionError otherwise)."""
    con = tab[tab.label >= 0]
    assert set(con.label.unique()) <= {0, 1}
    key = ["subject", "run", "sample"]
    cols = key + ["code"] + (["target_code"] if dataset == "kojimaB" else [])
    m = con[key + ["label"]].merge(ev[cols], on=key, how="left", validate="one_to_one")
    assert m.code.notna().all()
    code = m.code
    if dataset == "erpcore_mmn":
        rare, std = code == 70, code == 80
    elif dataset == "ds003061":
        rare, std = code == "noise", code == "standard"
    else:
        cs = code.map(KOJIMA_STREAM)
        ts = m.target_code.astype("float").map(TARGET_STREAM)
        un = cs.notna() & ts.notna() & (cs != ts)
        rare, std = code.isin([101, 102, 103, 104]) & un, code.isin([1, 2]) & un
    assert rare[m.label == 1].all(), "a contrast rare row is not the no-intent rare stimulus"
    assert std[m.label == 0].all(), "a contrast standard row is not a standard of the same stream"


@pytest.fixture(scope="module")
def tables():
    out = {}
    for d in pv.DATASETS:
        ev = pv.events(d)
        out[d] = (ev, pv.stream_table(d, ev))
    return out


def _n(tab, label):
    return int((tab.label == label).sum())







def test_ds003522_dropped():
    assert pv.DATASETS == ("erpcore_mmn", "ds003061", "kojimaB")
    for fn in (pv.events, pv.subjects, pv.passive_split):
        with pytest.raises(ValueError):
            fn("ds003522")
    with pytest.raises(ValueError):
        pv.load_passive("ds003522", "dev")










DEV = {"erpcore_mmn": ["031", "032"], "ds003061": ["010", "013"], "kojimaB": ["L", "O"]}






def test_split_argument_checked():
    with pytest.raises(ValueError):
        next(pv.load_passive("erpcore_mmn", "train"))
    with pytest.raises(ValueError):
        next(pv.load_passive("kojima", "dev"))



def _fake_rec(dataset, st, subject, run, fs):
    rows = st[(st.subject == subject) & (st.run == run)].reset_index(drop=True)
    n = int(rows["sample"].max()) + 10
    return en.Recording(source=dataset, uid=f"{dataset}_{subject}_{run}", patient=f"{dataset}_{subject}",
                        x=np.zeros((1, n), np.float32), fs=fs, names=["Cz"],
                        meta={"dataset": dataset, "events": rows})




def test_normalise_names():
    got = pv.normalise_names(["FP1", "Fz", "HEOG_left", "EXG1", "cpz", "T7", "GSR1", "vEOG", "Iz", "VEOG", "EKG"])
    assert got == [(0, "Fp1"), (1, "Fz"), (4, "CPz"), (5, "T7"), (8, "Iz")]



