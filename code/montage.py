"""Kit-8 positions in any montage.

Every Kit-8 position present by exact name (case-insensitive; old 10-20 names T3/T4/T5/T6 mapped to T7/T8/P7/P8)
is taken first. Each missing position then takes the nearest still-unused electrode by 3D distance in MNE's
standard_1005 montage. A montage with at most 8 usable channels is used whole."""
import numpy as np
import mne

KIT8 = ["Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8"]
ALIAS = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}
_POS = {}


def _positions():
    if not _POS:
        m = mne.channels.make_standard_montage("standard_1005")
        _POS.update({k.lower(): np.asarray(v) for k, v in m.get_positions()["ch_pos"].items()})
    return _POS


def canonical(name):
    n = str(name).strip()
    return ALIAS.get(n.upper(), n)


def kit8_indices(ch_names, exclude=()):
    pos = _positions()
    excl = {canonical(e).lower() for e in exclude}
    key = [canonical(c).lower() for c in ch_names]
    usable = [i for i, k in enumerate(key) if k in pos and k not in excl]
    if len(usable) <= 8:
        return usable
    out = [None] * 8
    for j, k in enumerate(KIT8):
        exact = [i for i in usable if key[i] == k.lower()]
        if exact:
            out[j] = exact[0]
    for j, k in enumerate(KIT8):
        if out[j] is None:
            free = [i for i in usable if i not in out]
            out[j] = free[int(np.argmin([np.linalg.norm(pos[key[i]] - pos[k.lower()]) for i in free]))]
    return out
