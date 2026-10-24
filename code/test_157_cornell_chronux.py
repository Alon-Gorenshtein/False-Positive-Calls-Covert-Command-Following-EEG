"""The Cornell two-group statistic (bench.pipe_cornell.tgt) against a literal transliteration of the public toolbox's
two_group_test_spectrum.m (Chronux 2.12; mirror https://raw.githubusercontent.com/jsiegle/chronux/
a3b39837d76309908f3ea91bb9feb72b2c749714/spectral_analysis/statistical_tests/two_group_test_spectrum.m, sha256
1608f946b7fc495c8ddce334e7596b8b2d3727dc0810d507e85c7ce8799c2ce6). The transliteration lives in 157_cornell_chronux
(chronux_two_group), loop for loop; nothing from the toolbox is imported or run."""
import importlib

import numpy as np
from scipy.special import polygamma

from bench import pipe_cornell as pc

cx = importlib.import_module("157_cornell_chronux")


def _inputs(i):
    """Power of tapered transforms, (n epochs, K tapers, ch, f) per condition, unequal group sizes."""
    g = np.random.default_rng([157, i])
    n1, n2, k, ch, nf = int(g.integers(6, 25)), int(g.integers(6, 25)), 5, 3, 31
    J1 = g.normal(size=(n1, k, ch, nf)) + 1j * g.normal(size=(n1, k, ch, nf))
    J2 = 1.2 * (g.normal(size=(n2, k, ch, nf)) + 1j * g.normal(size=(n2, k, ch, nf)))
    return (np.abs(J1) ** 2) * g.lognormal(0, 0.5, size=(n1, 1, 1, 1)), np.abs(J2) ** 2


def _toolbox(S1, S2):
    """Chronux on one (ch) slice at a time: J is frequencies x samples, samples = epochs x tapers."""
    out = [cx.chronux_two_group(np.sqrt(S1[:, :, c, :]).reshape(-1, S1.shape[-1]).T,
                                np.sqrt(S2[:, :, c, :]).reshape(-1, S2.shape[-1]).T) for c in range(S1.shape[2])]
    return tuple(np.stack([o[i] for o in out]) for i in range(3))


def test_dz_matches_toolbox():
    for i in range(30):
        S1, S2 = _inputs(i)
        dz, _ = pc.tgt(S1, S2)
        assert np.max(np.abs(dz - _toolbox(S1, S2)[0])) < 1e-10


def test_theoretical_decision_matches_toolbox():
    zc = 1.959963984540054
    for i in range(30):
        S1, S2 = _inputs(i)
        dz, _ = pc.tgt(S1, S2)
        assert np.array_equal((np.abs(dz) <= zc).astype(int), _toolbox(S1, S2)[2])


def test_toolbox_jackknife_reproduced_exactly():
    for i in range(30):
        S1, S2 = _inputs(i)
        assert np.max(np.abs(cx.tgt_toolbox(S1, S2)[1] - _toolbox(S1, S2)[1])) < 1e-10


def test_module_jackknife_differs_only_by_leave_one_out_scaling():
    """pipe_cornell divides each leave-one-out statistic by sqrt(psi'(m1) + psi'(m2)); the toolbox by
    sqrt(psi'(m1 - 1) + psi'(m2)) (group 1) and sqrt(psi'(m1) + psi'(m2 - 1)) (group 2)."""
    for i in range(30):
        S1, S2 = _inputs(i)
        _, v_mod = pc.tgt(S1, S2)
        _, v_tb, parts = cx.tgt_toolbox(S1, S2, parts=True)
        m1, m2 = S1.shape[0] * S1.shape[1], S2.shape[0] * S2.shape[1]
        sd2 = polygamma(1, m1) + polygamma(1, m2)
        rebuilt = parts[0] * (polygamma(1, m1 - 1) + polygamma(1, m2)) / sd2 + \
            parts[1] * (polygamma(1, m1) + polygamma(1, m2 - 1)) / sd2
        assert np.max(np.abs(rebuilt - v_mod)) < 1e-10
        assert np.all(v_mod >= v_tb)
