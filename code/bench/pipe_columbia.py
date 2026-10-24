"""Columbia CMD pipeline (Claassen NEJM 2019), EXACT: a thin wrapper around code/112_columbia_exact.py, which
reproduces the authors' public code bit for bit (the Columbia pipeline notes).
Nothing is re-implemented here.
- schedule: 112.schedule (6 blocks x 8 keep/stop trials, keep first, hands alternating by block, 3.2-s commands).
- run: 112.features + 112.exact_test (epoch-level permutation of labels and fold ids, 500, code rule p < 0.05).
- schedule_randomized / run_randomized: the same timing with keep/stop order flipped per trial (prob 0.5); the test
  is 112.trialflip_test, which re-draws exactly that per-trial flip (113_columbia_null test iii).
- Decision-exact early stopping at 112.STOP_AT for both tests (p of a stopped run is a lower bound, call exact).
- patient_rule: Benjamini-Hochberg over one patient's recordings, any rejection (112.bh_calls)."""
import numpy as np

from . import motor_common as mc

ce = mc.load_112()

NAME = 'columbia'
GRADE = 'EXACT'
FAMILY = 'cmd'
NATIVE_CH = list(ce.USE_CH)
RUN_S = 1505.0
N_PERM = 500
VARIANTS = {'as_coded': {'as_coded': True,
                         'cite': "Claassen 2019 and the authors' public code (code rule p < 0.05)"}}
N_TRIALS = ce.N_BLOCKS * ce.N_TRIALS
RESPONDER = {'resp_hand': {ce.KEEP_L: 'left', ce.STOP_L: 'none', ce.KEEP_R: 'right', ce.STOP_R: 'none'},
        'resp_delay_s': ce.CMD_S, 'resp_dur_s': ce.WIN_S}


def schedule(start_s, rng=None, variant='as_coded'):
    on, lab = ce.schedule(float(start_s))
    trial = np.repeat(np.arange(N_TRIALS), 2)
    return mc.Overlay(onsets_s=on, labels=lab, kind='cmd',
                      meta={'trial': trial, 'block': trial // ce.N_TRIALS, 'run': np.zeros(len(on), int),
                            'canonical': lab.copy(), 'flipped': np.zeros(N_TRIALS, bool), **RESPONDER})


def schedule_randomized(start_s, rng, variant='as_coded'):
    """Same onsets; within each trial the keep and stop commands swap places with probability 0.5."""
    ov = schedule(start_s)
    flip = mc.gen(rng).integers(0, 2, N_TRIALS).astype(bool)
    lab = ov.labels.reshape(-1, 2).copy()
    lab[flip] = lab[flip][:, ::-1]
    ov.labels, ov.meta['flipped'] = lab.ravel(), flip
    return ov


def extent(ov):
    """Absolute (t0, t1) seconds of the recording this overlay makes run/run_randomized epoch; must be clean
    (112.features filters the whole recording first)."""
    return float(ov.onsets_s.min()), float(ov.onsets_s.max()) + ce.CMD_S + ce.WIN_S + ce.TMAX


def _xy(rec, ov):
    """112.features on the overlay's timing (its canonical keep/stop ids drive their event shift), then the keep
    labels the overlay actually carries, 5 epochs per command."""
    canonical = np.asarray(ov.meta.get('canonical', ov.labels))
    X, _, g = mc.memo(mc.rec_key(rec, np.asarray(ov.onsets_s, float), canonical),
                      lambda: ce.features(rec.x, rec.fs, ov.onsets_s, canonical, rec.names))
    y = np.repeat(np.isin(np.asarray(ov.labels), (ce.KEEP_L, ce.KEEP_R)), ce.N_SEG)
    return X, y, g


def _pack(r):
    return mc.result(r['auc'], r['p'], r['call'], call_paper=r['call_paper'], se=r['se'],
                     stopped_early=r['stopped_early'], n_perm_run=r['n_perm_run'])


def run(rec, overlay, variant='as_coded', rng=None):
    X, y, g = _xy(rec, overlay)
    return _pack(ce.exact_test(X, y, g, ce._as_rs(rng), n_perm=N_PERM, stop_at=ce.STOP_AT))


def run_randomized(rec, overlay, variant='as_coded', rng=None):
    X, y, g = _xy(rec, overlay)
    return _pack(ce.trialflip_test(X, y, g, ce._as_rs(rng), n_perm=N_PERM, stop_at=ce.STOP_AT))


def stat_for_size_correction(result):
    return result['stat']


def patient_rule(results):
    return bool(np.any(ce.bh_calls([r['p'] for r in results]))) if results else False
