import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fretlib as F  # noqa: E402


def synth_trace(n=600, bleach=450, blink=(200, 215), bind=((100, 110), (300, 330)), alpha=0.09,
                gamma=0.7, d0=50000, noise=2500, seed=0):
    """Donor 50k; on binding E=0.4 with conserved gamma-corrected total."""
    rng = np.random.default_rng(seed)
    D = np.full(n, d0, float)
    Fa = np.zeros(n)                     # true acceptor signal
    for s, e in bind:
        D[s:e] = d0 * 0.6
        Fa[s:e] = gamma * d0 * 0.4
    A = Fa + alpha * D
    D, A = D + rng.normal(0, noise, n), A + rng.normal(0, noise, n)
    D[blink[0]:blink[1]] = rng.normal(0, noise, blink[1] - blink[0])
    A[blink[0]:blink[1]] = rng.normal(0, noise, blink[1] - blink[0])
    D[bleach:] = rng.normal(0, noise, n - bleach)
    A[bleach:] = rng.normal(0, noise, n - bleach)
    return D, A


def test_bleach_and_blink_detected():
    D, A = synth_trace()
    p = F.detect_photophysics(D, A)
    assert p.bleach_frame == 450
    assert len(p.blinks) == 1 and p.blinks[0] == (200, 215)
    assert not p.valid[450:].any() and not p.valid[200:215].any()
    assert p.valid[:190].all()                       # real frames kept, incl. binding events
    assert p.valid[100:110].all() and p.valid[300:330].all()


def test_no_bleach_trace_untouched():
    D, A = synth_trace(bleach=600, blink=(0, 0), bind=())
    p = F.detect_photophysics(D, A)
    assert p.status == "no_dark" and p.valid.all() and p.bleach_frame is None


def test_dead_trace_rejected():
    rng = np.random.default_rng(1)
    p = F.detect_photophysics(rng.normal(0, 1000, 300), rng.normal(0, 1000, 300))
    assert p.status == "no_signal" and not p.valid.any()


def test_leakage_and_gamma_recovered():
    traces, valids, events = [], [], []
    for i in range(8):
        D, A = synth_trace(seed=i, bind=((100, 110), (300 + 5 * i, 320 + 5 * i)))
        tr = F.Trace(f"t{i}", i, D, A)
        p = F.detect_photophysics(D, A)
        traces.append(tr); valids.append(p.valid)
        events.append(F.find_fret_events(D, A, p.valid))
    leak = F.estimate_leakage(traces, valids, events)
    assert leak["estimated"] and abs(leak["value"] - 0.09) < 0.01
    gam = F.estimate_gamma(traces, events, leak["value"])
    assert gam["estimated"] and abs(gam["value"] - 0.7) < 0.1


def test_defaults_when_data_insufficient():
    D, A = synth_trace(bind=())
    tr = F.Trace("t", 0, D, A)
    p = F.detect_photophysics(D, A)
    ev = F.find_fret_events(D, A, p.valid)
    leak = F.estimate_leakage([tr], [p.valid], [ev])
    gam = F.estimate_gamma([tr], [ev], leak["value"])
    assert not leak["estimated"] and leak["value"] == 0.09
    assert not gam["estimated"] and gam["value"] == 0.7


def test_dwell_fit_recovers_rate_with_censoring():
    rng = np.random.default_rng(0)
    k, dt = 0.2, 1.0
    t = rng.exponential(1 / k, 3000)
    n = np.ceil(t / dt).astype(int)
    cens = n > 30
    n = np.minimum(n, 30)
    r1 = F.fit_dwell_model(n, cens, dt, 1)
    assert abs(r1["rates"][0] - k) / k < 0.07


def test_two_component_preferred_for_mixture():
    rng = np.random.default_rng(1)
    t = np.where(rng.random(4000) < 0.5, rng.exponential(1 / 1.0, 4000), rng.exponential(1 / 0.05, 4000))
    n = np.maximum(1, np.ceil(t / 0.5).astype(int))
    c = np.zeros(n.size, bool)
    r1, r2 = F.fit_dwell_model(n, c, 0.5, 1), F.fit_dwell_model(n, c, 0.5, 2)
    assert r2["bic"] < r1["bic"] - 20
    assert abs(r2["rates"][0] - 0.05) / 0.05 < 0.25 and abs(r2["rates"][1] - 1.0) / 1.0 < 0.25


def test_extract_dwells_censoring_flags():
    b = np.array([0, 0, 1, 1, 1, 0, 0, 0], bool)
    d = F.extract_dwells([("x", 10, b)])
    assert d.n_frames.tolist() == [2, 3, 3]
    assert d.left_censored.tolist() == [True, False, False]
    assert d.right_censored.tolist() == [False, False, True]
    assert d.start_frame.tolist() == [10, 12, 15]


def test_global_and_individual_hmm_find_bound_state():
    rng = np.random.default_rng(0)
    seqs = []
    for _ in range(6):
        e = np.zeros(400)
        for s in rng.integers(20, 360, 3):
            e[s:s + rng.integers(5, 15)] = 0.5
        Dn = (1 - e) + rng.normal(0, 0.05, 400)
        An = e + rng.normal(0, 0.05, 400)
        seqs.append(np.column_stack([Dn, An]))
    X, L = np.vstack(seqs), [400] * 6
    g = F.fit_global_hmm(X, L, 3)
    E = F.state_fret(g.means_)
    assert E[0] < 0.1 and np.any(np.abs(E - 0.5) < 0.1)
    m, ok = F.fit_individual_hmm(seqs[0], [400], g)
    assert ok
    path = F.viterbi_segments(m, seqs[0], [400])
    assert (F.state_fret(m.means_)[path] > 0.3).sum() > 5


def test_gmm_selects_two_components():
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.normal(0.3, 0.04, 400), rng.normal(0.7, 0.04, 400)])
    g, tab = F.fit_gmm_1d(x)
    assert g.n_components == 2
