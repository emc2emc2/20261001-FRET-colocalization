"""Helper functions for the single-molecule two-colour TIRF (smFRET / colocalization) pipeline.

Conventions
-----------
* ``D``  = donor emission  under donor excitation  (``Dem-Dexc``)
* ``A``  = acceptor emission under donor excitation (``Aem-Dexc``)
* Leakage ``alpha``: fraction of the donor signal that bleeds into the acceptor channel,
  ``A_corr = A - alpha * D``.
* Gamma ``gamma``: detection-efficiency ratio, ``D_corr = gamma * D`` so that
  ``E = A_corr / (A_corr + gamma * D)`` and the total ``A_corr + gamma * D`` is conserved.
* Normalised signals: ``Dn = gamma*D / <total>``, ``An = A_corr / <total>`` with ``<total>``
  the mean of ``A_corr + gamma*D`` over the usable frames of that trace.  FRET of a point/state
  is ``An / (An + Dn)``.
"""
from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import ndimage, optimize

# --------------------------------------------------------------------------------------
# I/O
# --------------------------------------------------------------------------------------


@dataclass
class Trace:
    name: str
    pair: int
    D: np.ndarray
    A: np.ndarray
    movie: str = ""

    @property
    def n_frames(self) -> int:
        return len(self.D)


def read_trace(path) -> Trace:
    """Read one iSMS ``Traces_*_pair<N>.txt`` export (5 header lines, then two columns)."""
    path = Path(path)
    with open(path) as fh:
        header = [next(fh) for _ in range(5)]
    movie = ""
    pair = None
    for line in header:
        if line.lower().startswith("movie filename"):
            movie = line.split(":", 1)[1].strip()
        m = re.search(r"pair\s*#\s*(\d+)", line, flags=re.I)
        if m:
            pair = int(m.group(1))
    if pair is None:
        m = re.search(r"pair(\d+)", path.stem, flags=re.I)
        pair = int(m.group(1)) if m else -1
    data = np.loadtxt(path, skiprows=5, ndmin=2)
    return Trace(name=path.stem, pair=pair, D=data[:, 0].astype(float),
                 A=data[:, 1].astype(float), movie=movie)


def load_traces(directory, pattern="*.txt") -> list[Trace]:
    files = sorted(Path(directory).glob(pattern),
                   key=lambda p: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", p.name)])
    if not files:
        raise FileNotFoundError(f"No files matching {pattern!r} in {directory}")
    return [read_trace(f) for f in files]


# --------------------------------------------------------------------------------------
# Small utilities
# --------------------------------------------------------------------------------------


def find_runs(mask) -> list[tuple[int, int]]:
    """Half-open ``(start, end)`` index pairs of consecutive True runs."""
    m = np.concatenate([[0], np.asarray(mask, dtype=int), [0]])
    d = np.diff(m)
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def mad_sigma(x) -> float:
    x = np.asarray(x, float)
    return 1.4826 * np.median(np.abs(x - np.median(x)))


def two_means_1d(x, n_iter=100):
    """Deterministic 1-D 2-means (initialised at min/max). Returns (low_mean, high_mean)."""
    lo, hi = float(np.min(x)), float(np.max(x))
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        a, b = x[x <= mid], x[x > mid]
        if a.size == 0 or b.size == 0:
            break
        new_lo, new_hi = a.mean(), b.mean()
        if np.isclose(new_lo, lo) and np.isclose(new_hi, hi):
            break
        lo, hi = new_lo, new_hi
    return lo, hi


# --------------------------------------------------------------------------------------
# 1. Photobleaching / photoblinking detection
# --------------------------------------------------------------------------------------


@dataclass
class Photophysics:
    valid: np.ndarray                 # frames kept for analysis
    dark: np.ndarray                  # frames classified as dark (before padding)
    bleach_frame: Optional[int]
    blinks: list = field(default_factory=list)   # [(start, end_exclusive), ...]
    bright_level: float = np.nan
    dark_level: float = np.nan
    threshold: float = np.nan
    noise: float = np.nan
    status: str = "ok"                # ok | no_dark | no_signal


def detect_photophysics(D, A, *, dark_fraction=0.4, min_drop=0.5, min_bright_len=3,
                        min_bleach_len=5, pad=1, min_snr=8.0) -> Photophysics:
    """Detect photobleaching (terminal dark state) and photoblinking (transient dark states).

    Algorithm (acts on the total intensity ``T = D + A``, which is insensitive to FRET because
    donor loss is compensated by acceptor gain):

    1. Split ``T`` into a bright and a dark cluster with 1-D 2-means (deterministic init).
    2. The trace only has dark frames if the clusters are well separated: the dark level must be
       below ``(1 - min_drop)`` of the bright level and the bright level must exceed
       ``min_snr`` x the frame-to-frame noise (MAD of ``diff(T)``).  Otherwise nothing is flagged
       (``no_dark``) or the whole trace is rejected (``no_signal``).
    3. A frame is dark if ``T < dark_level + dark_fraction * (bright_level - dark_level)``.
       ``dark_fraction=0.4`` keeps high-FRET frames (whose total is reduced when gamma < 1).
    4. Bright runs shorter than ``min_bright_len`` frames (e.g. isolated spikes inside the
       post-bleach background) are reassigned to dark.
    5. The dark run that reaches the end of the trace (>= ``min_bleach_len`` frames) is the
       photobleaching event; every other dark run is a blink.
    6. Dark frames are dilated by ``pad`` frames to drop partially-averaged transition frames.
    """
    D = np.asarray(D, float)
    A = np.asarray(A, float)
    T = D + A
    n = len(T)
    dT = np.diff(T)
    noise = mad_sigma(dT) / np.sqrt(2) if n > 2 else np.nan

    lo, hi = two_means_1d(T)
    mid = 0.5 * (lo + hi)
    bright_level = float(np.median(T[T > mid])) if np.any(T > mid) else float(np.median(T))
    dark_level = float(np.median(T[T <= mid])) if np.any(T <= mid) else float(np.min(T))

    res = Photophysics(valid=np.ones(n, bool), dark=np.zeros(n, bool), bleach_frame=None,
                       bright_level=bright_level, dark_level=dark_level, noise=noise)
    if bright_level < min_snr * max(noise, 1e-9):
        res.status = "no_signal"
        res.valid[:] = False
        res.dark[:] = True
        return res
    if dark_level > (1 - min_drop) * bright_level:
        res.status = "no_dark"
        return res

    thr = dark_level + dark_fraction * (bright_level - dark_level)
    bright = T > thr
    for s, e in find_runs(bright):
        if e - s < min_bright_len:
            bright[s:e] = False
    dark = ~bright
    res.threshold = float(thr)
    res.dark = dark
    for s, e in find_runs(dark):
        if e == n and (e - s) >= min_bleach_len:
            res.bleach_frame = int(s)
        else:
            res.blinks.append((int(s), int(e)))
    padded = ndimage.binary_dilation(dark, iterations=pad) if pad > 0 else dark
    res.valid = ~padded
    return res


# --------------------------------------------------------------------------------------
# 2. Donor leakage and gamma
# --------------------------------------------------------------------------------------


@dataclass
class EventInfo:
    idx: np.ndarray        # frame indices of the valid frames (events are searched on these)
    rd: np.ndarray         # donor residual vs rolling median
    ra: np.ndarray         # acceptor residual vs rolling median
    cand: np.ndarray       # bool, frames that look like acceptor binding (donor down & acceptor up)
    event_id: np.ndarray   # 0 = none, 1.. = event number inside this trace


def find_fret_events(D, A, valid, *, window=101, n_sigma=3.0, merge_gap=1) -> EventInfo:
    """Model-free detection of anti-correlated donor-down / acceptor-up frames.

    Residuals are taken against a rolling median (robust to slow drift and to events shorter
    than half the window).  A frame is an event frame when the donor residual is below
    ``-n_sigma`` and the acceptor residual above ``+n_sigma`` robust standard deviations.
    """
    idx = np.flatnonzero(valid)
    if idx.size < 5:
        z = np.zeros(idx.size)
        return EventInfo(idx, z, z, np.zeros(idx.size, bool), np.zeros(idx.size, int))
    w = min(window, idx.size if idx.size % 2 else idx.size - 1)
    d, a = D[idx], A[idx]
    rd = d - ndimage.median_filter(d, size=w, mode="nearest")
    ra = a - ndimage.median_filter(a, size=w, mode="nearest")
    sd, sa = mad_sigma(rd), mad_sigma(ra)
    cand = (rd < -n_sigma * sd) & (ra > n_sigma * sa)
    lab, _ = ndimage.label(ndimage.binary_dilation(cand, iterations=merge_gap))
    return EventInfo(idx, rd, ra, cand, np.where(cand, lab, 0))


def estimate_leakage(traces, valids, events, *, pad=2, min_frames=100, min_traces=3,
                     default=0.09, plausible=(0.0, 0.5)) -> dict:
    """Donor leakage = acceptor/donor intensity ratio on frames with no acceptor bound.

    One ratio ``mean(A)/mean(D)`` is computed per trace on valid frames that are not within
    ``pad`` frames of a detected binding event; the global value is the median over traces.
    Falls back to ``default`` if fewer than ``min_traces`` traces have ``min_frames`` frames,
    or the estimate is implausible.
    """
    ratios = []
    for tr, valid, ev in zip(traces, valids, events):
        bound = np.zeros(tr.n_frames, bool)
        bound[ev.idx[ev.cand]] = True
        bound = ndimage.binary_dilation(bound, iterations=pad)
        unb = valid & ~bound
        if unb.sum() >= min_frames and tr.D[unb].mean() > 0:
            ratios.append(tr.A[unb].mean() / tr.D[unb].mean())
    ratios = np.asarray(ratios)
    out = dict(per_trace=ratios, n_traces=len(ratios), default=default)
    if len(ratios) >= min_traces:
        val = float(np.median(ratios))
        if plausible[0] <= val <= plausible[1]:
            out.update(value=val, estimated=True, iqr=tuple(np.percentile(ratios, [25, 75])))
            return out
        warnings.warn(f"Estimated leakage {val:.3f} is implausible; using default {default}.")
    out.update(value=default, estimated=False, iqr=(np.nan, np.nan))
    return out


def estimate_gamma(traces, events, alpha, *, min_events=5, default=0.7, plausible=(0.2, 3.0),
                   n_boot=1000, seed=0) -> dict:
    """Gamma from the anti-correlated donor/acceptor change at acceptor binding events.

    For every event frame: ``x = -dD`` (donor loss) and ``y = dA - alpha*dD`` (leakage-corrected
    acceptor gain) relative to the local rolling-median baseline.  Conservation of the total
    ``A_corr + gamma*D`` implies ``y = gamma * x``; gamma is the slope through the origin
    ``sum(xy)/sum(x^2)`` over all event frames of all traces.  A bootstrap over events gives a CI.
    Falls back to ``default`` with fewer than ``min_events`` events or an implausible value.
    """
    xs, ys, keys = [], [], []
    for i, ev in enumerate(events):
        m = ev.cand
        xs.append(-ev.rd[m])
        ys.append(ev.ra[m] - alpha * ev.rd[m])
        keys.extend((i, k) for k in ev.event_id[m])
    x = np.concatenate(xs) if xs else np.empty(0)
    y = np.concatenate(ys) if ys else np.empty(0)
    keys = np.array(keys) if keys else np.empty((0, 2), int)
    uniq, inv = np.unique(keys, axis=0, return_inverse=True) if len(keys) else (np.empty((0, 2)), np.empty(0, int))
    n_ev = len(uniq)
    out = dict(n_events=n_ev, n_frames=len(x), default=default, x=x, y=y)
    if n_ev:
        out["per_event"] = np.array([y[inv == e].sum() / x[inv == e].sum() for e in range(n_ev)])
    if n_ev >= min_events:
        val = float((x * y).sum() / (x * x).sum())
        rng = np.random.default_rng(seed)
        boots = []
        for _ in range(n_boot):
            pick = rng.integers(0, n_ev, n_ev)
            sel = np.concatenate([np.flatnonzero(inv == e) for e in pick])
            boots.append((x[sel] * y[sel]).sum() / (x[sel] ** 2).sum())
        if plausible[0] <= val <= plausible[1]:
            out.update(value=val, estimated=True, ci=tuple(np.percentile(boots, [2.5, 97.5])))
            return out
        warnings.warn(f"Estimated gamma {val:.3f} is implausible; using default {default}.")
    out.update(value=default, estimated=False, ci=(np.nan, np.nan))
    return out


# --------------------------------------------------------------------------------------
# 3. Correction and normalisation
# --------------------------------------------------------------------------------------


def correct_and_normalize(D, A, valid, alpha, gamma):
    """Return ``Dn, An, mean_total`` (see module docstring)."""
    FA = A - alpha * D
    FD = gamma * D
    mean_total = float((FA + FD)[valid].mean())
    return FD / mean_total, FA / mean_total, mean_total


def split_segments(valid, min_len=5) -> list[tuple[int, int]]:
    """Contiguous runs of valid frames (>= min_len); HMMs never bridge excluded frames."""
    return [(int(s), int(e)) for s, e in find_runs(valid) if e - s >= min_len]


def frame_fret(Dn, An):
    tot = An + Dn
    return np.divide(An, tot, out=np.full_like(tot, np.nan), where=tot > 1e-9)


# --------------------------------------------------------------------------------------
# 4. Gaussian HMM on (Dn, An)
# --------------------------------------------------------------------------------------


def state_fret(means) -> np.ndarray:
    means = np.asarray(means)
    return means[:, 1] / (means[:, 0] + means[:, 1])


def init_hmm_params(X, n_states=5, e_split=0.15):
    """Data-driven starting values.

    State 0 is the acceptor-free (donor-only) cloud: frames with frame FRET < ``e_split``.  The
    other states are placed at evenly spaced quantiles of the FRET distribution of the remaining
    frames, because binding events are rare and plain k-means would put all centres inside the
    donor-only cloud.
    """
    E = frame_fret(X[:, 0], X[:, 1])
    ok = np.isfinite(E)
    unb = ok & (E < e_split)
    if unb.sum() < 20:
        unb = ok
    mu0 = np.median(X[unb], axis=0)
    cov0 = np.cov(X[unb].T) + 1e-6 * np.eye(2)
    s = float(np.median(X[unb].sum(axis=1)))
    nb = n_states - 1
    Eb = E[ok & (E >= e_split) & (E < 1.0)]
    if Eb.size >= 10 * nb:
        levels = np.quantile(Eb, (np.arange(nb) + 0.5) / nb)
    else:
        levels = np.linspace(0.25, 0.85, nb)
    means = np.vstack([mu0] + [[(1 - e) * s, e * s] for e in levels])
    covars = np.stack([cov0] + [1.5 * cov0] * nb)
    start = np.full(n_states, 0.1 / nb)
    start[0] = 0.9
    trans = np.full((n_states, n_states), 0.02 / (n_states - 1))
    np.fill_diagonal(trans, 0.98)
    return start, trans, means, covars


def _make_hmm(n_states, start, trans, means, covars, *, n_iter, tol, min_covar, **priors):
    from hmmlearn.hmm import GaussianHMM

    m = GaussianHMM(n_components=n_states, covariance_type="full", n_iter=n_iter, tol=tol,
                    min_covar=min_covar, init_params="", params="stmc", **priors)
    m.startprob_ = np.asarray(start, float)
    m.transmat_ = np.asarray(trans, float)
    m.means_ = np.asarray(means, float)
    m.covars_ = np.asarray(covars, float)
    return m


def sort_states_by_fret(model):
    """Reorder states in place by ascending FRET."""
    order = np.argsort(state_fret(model.means_))
    model.startprob_ = model.startprob_[order]
    model.transmat_ = model.transmat_[np.ix_(order, order)]
    model.means_ = model.means_[order]
    model.covars_ = model.covars_[order]
    return model


def fit_global_hmm(X, lengths, n_states=5, *, n_iter=200, tol=1e-4, min_covar=1e-5, e_split=0.15):
    """Fit one HMM jointly to all usable segments of all traces (full-covariance 2-D Gaussians)."""
    start, trans, means, covars = init_hmm_params(X, n_states, e_split)
    m = _make_hmm(n_states, start, trans, means, covars, n_iter=n_iter, tol=tol,
                  min_covar=min_covar, covars_prior=0.0, covars_weight=0.0)
    m.fit(X, lengths)
    sort_states_by_fret(m)
    m.converged_ = bool(m.monitor_.converged)
    m.loglik_ = float(m.score(X, lengths))
    return m


def fit_individual_hmm(X, lengths, g, *, prior_strength=10.0, n_iter=50, tol=1e-3, min_covar=1e-5):
    """Re-fit one trace starting from the global model ``g``.

    ``prior_strength`` adds a weak MAP prior centred on the global parameters, worth that many
    pseudo-frames/pseudo-transitions.  It does not change well-populated states but stops
    states that never occur in this trace from drifting onto unrelated data (0 = pure ML).
    Falls back to the global model if the fit fails.  Returns ``(model, ok)``.
    """
    K, d = g.means_.shape
    w = float(prior_strength)
    priors = {}
    if w > 0:
        priors = dict(means_prior=g.means_.copy(), means_weight=w,
                      covars_prior=w * g.covars_.copy(), covars_weight=d + w,
                      startprob_prior=1 + w * g.startprob_, transmat_prior=1 + w * g.transmat_)
    else:
        priors = dict(covars_prior=0.0, covars_weight=0.0)
    m = _make_hmm(K, g.startprob_, g.transmat_, g.means_, g.covars_, n_iter=n_iter, tol=tol,
                  min_covar=min_covar, **priors)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.fit(X, lengths)
        ok = all(np.all(np.isfinite(v)) for v in (m.startprob_, m.transmat_, m.means_, m.covars_))
        if ok:
            np.linalg.cholesky(m.covars_)  # raises if not positive definite
    except Exception:
        ok = False
    if not ok:
        return g, False
    return m, True


def viterbi_segments(model, X, lengths) -> np.ndarray:
    """Viterbi state path for every segment (concatenated)."""
    paths, pos = [], 0
    for L in lengths:
        paths.append(model.predict(X[pos:pos + L]))
        pos += L
    return np.concatenate(paths) if paths else np.empty(0, int)


def state_table(model, e_thr=None) -> pd.DataFrame:
    E = state_fret(model.means_)
    df = pd.DataFrame(dict(state=np.arange(len(E)), Dn=model.means_[:, 0], An=model.means_[:, 1],
                           FRET=E, sd_Dn=np.sqrt(model.covars_[:, 0, 0]),
                           sd_An=np.sqrt(model.covars_[:, 1, 1]),
                           self_transition=np.diag(model.transmat_)))
    if e_thr is not None:
        df["bound"] = df.FRET > e_thr
    return df


# --------------------------------------------------------------------------------------
# 5. Dwell times
# --------------------------------------------------------------------------------------


def extract_dwells(bound_paths) -> pd.DataFrame:
    """Dwell table from ``[(trace_name, seg_start, bool_array), ...]``.

    A dwell touching the start of a segment is left-censored (true start unknown) and one
    touching the end is right-censored (true end unknown: bleaching, blink or end of movie).
    """
    rows = []
    for name, start, b in bound_paths:
        n = len(b)
        change = np.flatnonzero(np.diff(b.astype(int))) + 1
        edges = np.concatenate([[0], change, [n]])
        for s, e in zip(edges[:-1], edges[1:]):
            rows.append(dict(trace=name, start_frame=start + int(s), n_frames=int(e - s),
                             bound=bool(b[s]), left_censored=(s == 0), right_censored=(e == n)))
    return pd.DataFrame(rows)


def _survival(t, k, w):
    t = np.asarray(t, float)[:, None]
    return (w[None, :] * np.exp(-k[None, :] * t)).sum(axis=1)


def _unpack(theta, n_comp):
    k = np.exp(theta[:n_comp])
    if n_comp == 1:
        return k, np.array([1.0])
    z = np.concatenate([[0.0], theta[n_comp:]])
    w = np.exp(z - z.max())
    return k, w / w.sum()


def _nll(theta, n_comp, n, cens, dt):
    k, w = _unpack(theta, n_comp)
    s_prev = _survival((n - 1) * dt, k, w)
    s_cur = _survival(n * dt, k, w)
    p = np.where(cens, s_prev, s_prev - s_cur)
    return -np.sum(np.log(np.maximum(p, 1e-300)))


def fit_dwell_model(n_frames, censored, dt=1.0, n_comp=1, n_starts=30, seed=0) -> dict:
    """Maximum-likelihood fit of a 1- or 2-component exponential (rate) model.

    Dwells are integer numbers of frames: an observed dwell of ``n`` frames has true duration in
    ``((n-1)dt, n dt]`` (probability ``S((n-1)dt) - S(n dt)``); a right-censored dwell of ``n``
    frames contributes ``S((n-1)dt)``.  ``S(t) = sum_i w_i exp(-k_i t)``.
    """
    n = np.asarray(n_frames, int)
    cens = np.asarray(censored, bool)
    if n.size == 0:
        raise ValueError("no dwells to fit")
    rng = np.random.default_rng(seed)
    kmax = 10.0 / dt
    kmin = 1e-3 / (dt * max(n.sum(), 1))
    bounds = [(np.log(kmin), np.log(kmax))] * n_comp + [(-8, 8)] * (n_comp - 1)
    mean_rate = max((n.sum() - 0.5 * n.size) * dt, 0.5 * dt)
    k0 = n.size / mean_rate
    best = None
    for i in range(1 if n_comp == 1 else n_starts):
        if n_comp == 1:
            theta0 = [np.log(np.clip(k0, kmin, kmax))]
        else:
            theta0 = list(np.log(np.sort(k0 * 10 ** rng.uniform(-1.5, 1.5, 2)))) + [rng.normal(0, 1)]
        r = optimize.minimize(_nll, theta0, args=(n_comp, n, cens, dt), method="L-BFGS-B", bounds=bounds)
        if best is None or r.fun < best.fun:
            best = r
    k, w = _unpack(best.x, n_comp)
    order = np.argsort(k)
    k, w = k[order], w[order]
    n_par = 2 * n_comp - 1
    ll = -best.fun
    return dict(n_comp=n_comp, rates=k, weights=w, loglik=ll, n_par=n_par,
                aic=2 * n_par - 2 * ll, bic=n_par * np.log(n.size) - 2 * ll,
                mean_dwell=float((w / k).sum()), n_dwells=int(n.size), n_censored=int(cens.sum()),
                dt=dt)


def bootstrap_dwell_model(n_frames, censored, dt, n_comp, n_boot=200, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = np.asarray(n_frames)
    c = np.asarray(censored)
    rows = []
    for b in range(n_boot):
        pick = rng.integers(0, n.size, n.size)
        try:
            r = fit_dwell_model(n[pick], c[pick], dt, n_comp, n_starts=6, seed=b)
        except Exception:
            continue
        rows.append(dict(**{f"k{i + 1}": k for i, k in enumerate(r["rates"])},
                         **{f"w{i + 1}": w for i, w in enumerate(r["weights"])},
                         mean_dwell=r["mean_dwell"]))
    return pd.DataFrame(rows)


def model_survival_curve(fit, t):
    return _survival(t, fit["rates"], fit["weights"])


def kaplan_meier(n_frames, censored, dt=1.0):
    """Kaplan-Meier survival estimate for frame-quantised, right-censored dwells."""
    n = np.asarray(n_frames, int)
    c = np.asarray(censored, bool)
    times = np.unique(n[~c])
    surv, s = [], 1.0
    for t in times:
        at_risk = np.sum(n >= t)
        d = np.sum((n == t) & ~c)
        s *= 1 - d / at_risk
        surv.append(s)
    return times * dt, np.array(surv)


# --------------------------------------------------------------------------------------
# 6. FRET histogram / Gaussian mixture
# --------------------------------------------------------------------------------------


def fit_gmm_1d(values, max_components=4, n_components=None, seed=0, min_per_component=10):
    """1-D Gaussian mixture; number of components chosen by BIC unless given.

    Returns ``(best_model_or_None, table_of_BIC)``.
    """
    from sklearn.mixture import GaussianMixture

    x = np.asarray(values, float)
    x = x[np.isfinite(x)].reshape(-1, 1)
    if x.shape[0] < 3:
        return None, pd.DataFrame()
    cap = max(1, x.shape[0] // min_per_component)
    cands = [n_components] if n_components else range(1, min(max_components, cap) + 1)
    rows, models = [], {}
    for k in cands:
        g = GaussianMixture(k, covariance_type="full", n_init=5, random_state=seed, reg_covar=1e-6).fit(x)
        models[k] = g
        rows.append(dict(n_components=k, BIC=g.bic(x), AIC=g.aic(x), loglik=g.score(x) * len(x)))
    tab = pd.DataFrame(rows)
    best = models[int(tab.loc[tab.BIC.idxmin(), "n_components"])]
    return best, tab
