"""Tests for 112_columbia_exact.py (the Columbia pipeline notes).

(a) Equivalence with the authors' own functions (kwdoyle/CMD-Detection @ 38d013c, eeg_functions.read_data and
    run_pipeline with cmd_detection.py's arguments), imported by path from output/external/ inside the tests only:
    a2 pinned: their code under their requirements.txt versions (mne 1.1.1, scikit-learn 1.2.0, numpy 1.23.3, pandas
       1.5.0, pyriemann 0.3; scipy 1.9.1 assumed), saved by --make-ref to output/intermediate/columbia_exact_ref.npz;
       this interpreter's features/exact_test must match it (features to rtol 1e-9; fold AUCs, AUC, p and every
       permutation AUC exactly).
    a1 live: their code under this interpreter (mne 1.8, scikit-learn 1.6.1). Three things are restored to the pinned
       behaviour: pyriemann is stubbed (not used by the 'psd' branch), mne's removed psd_multitaper is supplied from
       112's MNE 1.1.1 transcription, and LinearSVC gets dual=True (the 1.2.0 default; 1.6.1 defaults to 'auto', which
       picks the primal solver here). Everything else (events, shift, epoching, pycsd CSD, bands, CV, shuffles, p) is
       their code.
(b) schedule timing; (c) shapes; (d) the trial-flip null keeps runs and folds.
/usr/bin/python3 -I code/test_columbia_exact.py               all tests
<pinned venv python> -I code/test_columbia_exact.py --make-ref  writes the a2 fixture (only the authors' code runs)"""
import argparse
import contextlib
import importlib.util
import io
import os
import sys
import tempfile
import time
import types
import warnings

if '--make-ref' not in sys.argv:
    import site
    sys.path.append(site.getusersitepackages())
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CMD_DIR = os.path.join(ROOT, 'output', 'external', 'cmd_detection',
                       'CMD-Detection-38d013c58b1423ba05fd1cd0c2417d0beb380671')
PYCSD_DIR = os.path.join(ROOT, 'output', 'external', 'pycsd', 'pycsd-2d06d2f9016297d4163e2ca73ed04319ff2d023b')
REF = os.path.join(ROOT, 'output', 'intermediate', 'columbia_exact_ref.npz')
SCEN = (('null128', 128.0, 0.0, 100, 11), ('signal128', 128.0, 0.5, 100, 12), ('null200', 200.0, 0.0, 30, 13))
LIVE = (('null128', 128.0, 0.0, 40, 21), ('signal128', 128.0, 0.5, 40, 22))


def _load(name, path, search=None):
    spec = importlib.util.spec_from_file_location(name, path, submodule_search_locations=search)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_ce():
    return _load('columbia_exact', os.path.join(HERE, '112_columbia_exact.py'))


def load_theirs(ce=None):
    """Their eeg_functions.py and the pycsd it imports, by path. ce given -> modern interpreter: apply the a1 shims."""
    _load('pycsd', os.path.join(PYCSD_DIR, 'pycsd', '__init__.py'), [os.path.join(PYCSD_DIR, 'pycsd')])
    if ce is not None:
        class _Unused:
            def __init__(self, *a, **k):
                raise RuntimeError('pyriemann stub: not used by the psd branch')
        pr, ts, es = (types.ModuleType(n) for n in ('pyriemann', 'pyriemann.tangentspace', 'pyriemann.estimation'))
        ts.TangentSpace = _Unused
        es.Covariances = es.CospCovariances = es.HankelCovariances = _Unused
        pr.tangentspace, pr.estimation = ts, es
        sys.modules.update({'pyriemann': pr, 'pyriemann.tangentspace': ts, 'pyriemann.estimation': es})
        import mne.time_frequency

        def psd_multitaper(inst, fmin=0, fmax=np.inf):
            return ce.psd_multitaper_111(inst.get_data(picks='data'), inst.info['sfreq'], fmin, fmax)
        mne.time_frequency.psd_multitaper = psd_multitaper
    ef = _load('cmd_eeg_functions', os.path.join(CMD_DIR, 'eeg_functions.py'))
    if ce is not None:
        from sklearn.svm import LinearSVC

        def linear_svc_120(**kw):
            return LinearSVC(dual=True, **kw)
        ef.LinearSVC = linear_svc_120
    return ef


def synth(ce, fs, effect, seed):
    """19-channel µV recording with the schedule overlaid: AR(1) background, a slowly modulated 10-Hz rhythm and,
    with effect > 0, the rhythm at C3/Cz/C4 reduced by `effect` in the 10 s after each keep command."""
    from scipy.signal import lfilter
    rs = np.random.RandomState(seed)
    on, lab = ce.schedule(20.0)
    n = int(round((on[-1] + ce.CMD_S + ce.WIN_S + 20.0) * fs))
    t = np.arange(n) / fs
    x = lfilter([1.0], [1.0, -0.95], rs.standard_normal((19, n)) * 6.0, axis=1)
    amp = 8.0 * (1 + 0.4 * np.sin(2 * np.pi * t / 37.0 + rs.uniform(0, 2 * np.pi, (19, 1))))
    alpha = amp * np.sin(2 * np.pi * 10.0 * t + rs.uniform(0, 2 * np.pi, (19, 1)))
    if effect:
        gain = np.ones(n)
        for o in on[np.isin(lab, (ce.KEEP_L, ce.KEEP_R))] + ce.CMD_S:
            gain[int(o * fs):int((o + ce.WIN_S) * fs)] = 1 - effect
        for c in ('C3', 'Cz', 'C4'):
            alpha[ce.USE_CH.index(c)] *= gain
    return x + alpha + rs.standard_normal((19, n)), on, lab


def run_theirs(ef, x_uV, fs, on, lab, n_perm, seed, use_ch):
    """cmd_detection.main for one file: read_data(...) then run_pipeline(...) with its parameters (l. 43-55, 100-139).
    Returns their (score, se, pvalue, perm) and the X, y, groups and fold AUCs seen by their first cross_val_score."""
    import mne
    mne.utils.set_log_level('ERROR')
    stim = np.zeros(x_uV.shape[1])
    for s, i in zip(np.round(on * fs).astype(int), lab):
        stim[s:s + 5] = i
    info = mne.create_info(list(use_ch) + ['STI 014'], fs, ['eeg'] * len(use_ch) + ['stim'])
    raw = mne.io.RawArray(np.vstack([x_uV * 1e-6, stim]), info)
    rec = {}
    real_cvs = ef.cross_val_score

    def cvs(clf, X, y, **kw):
        out = real_cvs(clf, X, y, **kw)
        if not rec:
            rec.update(X=np.array(X), y=np.asarray(y, bool), groups=np.array(kw['groups']), scores=np.array(out))
        return out
    ef.cross_val_score = cvs
    with tempfile.TemporaryDirectory() as td, warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.filterwarnings('ignore')
        fname = os.path.join(td, 'synth-raw.fif')
        raw.save(fname, fmt='double', overwrite=True)
        epochs = ef.read_data(data=fname, event_fl=[], use_ch=list(use_ch), tmin=0.0, tmax=2.0, fmin=1.0, fmax=30.0,
                              n_epo_segments=5, hand_use=None, is_control=False)
        assert len(set(len(epochs[e]) for e in epochs.event_id)) == 1
        n_times = len(epochs.times)
        np.random.seed(seed)
        out = ef.run_pipeline(epochs=epochs, fmin=1.0, fmax=30.0, pipe_type='psd', overlap=0.9, delays=[1, 2, 4, 8],
                              bands=((1, 3), (4, 7), (8, 13), (14, 30)), n_permutations=n_perm, laplac_ref=True)
    ef.cross_val_score = real_cvs
    return out, rec, n_times


def make_ref():
    import mne
    import sklearn
    import pandas
    import scipy
    ce = load_ce()
    ef = load_theirs()
    vers = f'mne {mne.__version__}, sklearn {sklearn.__version__}, numpy {np.__version__}, pandas ' \
           f'{pandas.__version__}, scipy {scipy.__version__}, python {sys.version.split()[0]}'
    print(vers)
    out = {'versions': np.array(vers)}
    for name, fs, effect, n_perm, seed in SCEN:
        t0 = time.time()
        x, on, lab = synth(ce, fs, effect, seed)
        (score, se, p, perm), rec, n_times = run_theirs(ef, x, fs, on, lab, n_perm, seed, ce.USE_CH)
        out.update({f'{name}_{k}': v for k, v in rec.items()})
        out.update({f'{name}_x': x, f'{name}_on': on, f'{name}_lab': lab, f'{name}_auc': score, f'{name}_se': se,
                    f'{name}_p': p, f'{name}_perm': np.array(perm), f'{name}_ntimes': n_times})
        print(f'{name}: auc {score:.4f} p {p:.4f} X {rec["X"].shape} n_times {n_times} ({time.time() - t0:.0f} s)')
    os.makedirs(os.path.dirname(REF), exist_ok=True)
    np.savez_compressed(REF, **out)
    print('->', REF)



def _same_run(ce, mine, X, y, groups, scores, auc, se, p, perm, tag):
    np.testing.assert_allclose(mine['features'], X, rtol=1e-9, atol=0, err_msg=f'{tag} features')
    assert (mine['y'] == y).all() and (mine['groups'] == groups).all(), f'{tag} labels/groups'
    r = mine['res']
    assert np.array_equal(mine['scores'], scores), f'{tag} fold AUCs {np.abs(mine["scores"] - scores).max()}'
    assert r['auc'] == auc and r['p'] == p and np.isclose(r['se'], se, rtol=1e-12), f'{tag} auc/p/se'
    assert np.array_equal(r['perm'], np.asarray(perm)), f'{tag} perm AUCs differ at {np.flatnonzero(r["perm"] != perm)}'
    return f'auc {auc:.4f} p {p:.4f} ({len(perm)} perms, max |dX|/|X| ' \
           f'{np.max(np.abs(mine["features"] - X) / np.abs(X)):.1e})'


def _mine(ce, x, fs, on, lab, seed, n_perm):
    X, y, g = ce.features(x, fs, on, lab, ce.USE_CH)
    scores = ce.cv_scores(X, y, g, np.random.RandomState(seed))
    res = ce.exact_test(X, y, g, np.random.RandomState(seed), n_perm=n_perm)
    return {'features': X, 'y': y, 'groups': g, 'scores': scores, 'res': res}


def test_a2_pinned(ce):
    if not os.path.exists(REF):
        return 'SKIP: no fixture; run --make-ref under the pinned venv'
    z = np.load(REF)
    msgs = [str(z['versions'])]
    for name, fs, effect, n_perm, seed in SCEN:
        mine = _mine(ce, z[f'{name}_x'], fs, z[f'{name}_on'], z[f'{name}_lab'], seed, n_perm)
        assert int(z[f'{name}_ntimes']) == int(round(2 * fs)) + 1
        msgs.append(name + ': ' + _same_run(ce, mine, z[f'{name}_X'], z[f'{name}_y'], z[f'{name}_groups'],
                                            z[f'{name}_scores'], float(z[f'{name}_auc']), float(z[f'{name}_se']),
                                            float(z[f'{name}_p']), z[f'{name}_perm'], name))
    return '; '.join(msgs)


def test_a1_live(ce):
    ef = load_theirs(ce)
    csd = sys.modules['pycsd.csd']
    pos = csd._extract_positions(types.SimpleNamespace(info={'description': 'standard/1020'}, ch_names=ce.USE_CH),
                                 None)
    Gi, TC, sgi = csd._prepare_G(csd._calc_g(np.dot(pos, pos.T)), 1e-5)
    mGi, mTC, msgi, mH = ce.csd_matrices()
    assert np.array_equal(Gi, mGi) and np.array_equal(TC, mTC) and sgi == msgi
    assert np.array_equal(csd._calc_h(np.dot(pos, pos.T)), mH)
    msgs = ['CSD matrices identical to pycsd']
    for name, fs, effect, n_perm, seed in LIVE:
        x, on, lab = synth(ce, fs, effect, seed)
        (auc, se, p, perm), rec, _ = run_theirs(ef, x, fs, on, lab, n_perm, seed, ce.USE_CH)
        mine = _mine(ce, x, fs, on, lab, seed, n_perm)
        msgs.append(name + ': ' + _same_run(ce, mine, rec['X'], rec['y'], rec['groups'], rec['scores'], auc, se, p,
                                            perm, name))
    return '; '.join(msgs)


def test_b_schedule(ce):
    on, lab = ce.schedule(100.0)
    assert len(on) == 96 and on[0] == 100.0
    keep = np.isin(lab, (ce.KEEP_L, ce.KEEP_R))
    assert (keep[0::2]).all() and (~keep[1::2]).all()
    hand = np.where(np.isin(lab, (ce.KEEP_R, ce.STOP_R)), 'R', 'L').reshape(6, 16)
    assert [h[0] for h in hand] == list('RLRLRL') and all((h == h[0]).all() for h in hand)
    d = np.diff(on)
    assert np.allclose(d[0::2], 3.2 + 10.0) and np.allclose(d[1::2], 3.2 + 15.0)
    assert np.isclose(on[-1] + 3.2 + 10.0 - on[0], 1502.2)
    on2, _ = ce.schedule(0.0, block_gap_s=40.0, first_hand='left')
    assert np.isclose(on2[16] - on2[15], 3.2 + 40.0) and np.isclose(on2[17] - on2[16], 13.2)
    for fs in (128.0, 200.0, 256.0, 512.0):
        start, ids, g = ce.epoch_events(on, lab, fs)
        offs = start[0::5]
        assert np.abs(offs - np.round((on + 3.2) * fs)).max() <= 1, fs
        assert (np.diff(start.reshape(96, 5), axis=1) == int(2 * fs)).all()
    return 'six 16-command blocks R/L alternating; 13.2 s keep->stop, 18.2 s stop->keep; span 1502.2 s; ' \
           'shift_events recovers offsets within 1 sample at 128/200/256/512 Hz'


def test_c_shapes(ce):
    x, on, lab = synth(ce, 128.0, 0.0, 5)
    X, y, g = ce.features(x, 128.0, on, lab, ce.USE_CH)
    assert X.shape == (480, 76) and y.shape == g.shape == (480,) and y.sum() == 240
    assert (np.unique(g) == np.arange(48)).all() and all((g == k).sum() == 10 and y[g == k].sum() == 5 for k in range(48))
    assert (y.reshape(96, 5) == y.reshape(96, 5)[:, :1]).all() and (y[::10]).all()
    assert np.isfinite(X).all() and (X > 0).all()
    old = [{'T7': 'T3', 'T8': 'T4', 'P7': 'T5', 'P8': 'T6'}.get(c, c) for c in ce.USE_CH][::-1]
    X2, _, _ = ce.features(x[::-1], 128.0, on, lab, old)
    assert np.array_equal(X2, X)
    res = ce.exact_test(X, y, g, np.random.RandomState(0), n_perm=3)
    assert res['perm'].shape == (3,) and 0 < res['p'] <= 1 and set(res) >= {'auc', 'p', 'call'}
    return 'X (480, 76), 48 trial folds of 5 keep + 5 stop epochs, old 10-20 names and any row order accepted'


def test_d_trialflip(ce):
    x, on, lab = synth(ce, 128.0, 0.0, 6)
    _, y, g = ce.features(x, 128.0, on, lab, ce.USE_CH)
    rs = np.random.RandomState(3)
    flips = []
    for _ in range(200):
        yp = ce.trialflip_labels(y, g, rs)
        runs = yp.reshape(96, 5)
        assert (runs == runs[:, :1]).all()
        assert (runs[0::2, 0] != runs[1::2, 0]).all()
        f = (yp != y).reshape(48, 10)
        assert (f == f[:, :1]).all()
        flips.append(f[:, 0].mean())
    assert 0.45 < np.mean(flips) < 0.55
    order = list(range(len(y)))
    rs.shuffle(order)
    assert not (y[order].reshape(96, 5) == y[order].reshape(96, 5)[:, :1]).all()
    r = ce.trialflip_test(*ce.features(x, 128.0, on, lab, ce.USE_CH), np.random.RandomState(0), n_perm=3)
    assert r['perm'].shape == (3,)
    return f'200 nulls: every 5-epoch run intact, keep/stop opposite within trial, mean flip share ' \
           f'{np.mean(flips):.3f}; folds untouched (groups not permuted); the epoch shuffle breaks runs'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--make-ref', action='store_true')
    ap.add_argument('--only', default='')
    a = ap.parse_args()
    if a.make_ref:
        return make_ref()
    ce = load_ce()
    fails = 0
    for t in (test_b_schedule, test_c_shapes, test_d_trialflip, test_a2_pinned, test_a1_live):
        if a.only and a.only not in t.__name__:
            continue
        t0 = time.time()
        try:
            msg = t(ce)
            print(f'PASS {t.__name__} ({time.time() - t0:.0f} s): {msg}')
        except AssertionError as e:
            fails += 1
            print(f'FAIL {t.__name__}: {e}')
    sys.exit(1 if fails else 0)


if __name__ == '__main__':
    main()
