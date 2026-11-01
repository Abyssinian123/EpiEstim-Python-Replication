# -*- coding: utf-8 -*-
"""
Pure-NumPy port of the methods used in

  Nash RK, Bhatt S, Cori A, Nouvellet P (2023). Estimating the epidemic
  reproduction number from temporally aggregated incidence data.
  PLoS Comput Biol 19(8): e1011439.

Faithful to the R implementation in the EpiEstim package (mrc-ide/EpiEstim):
  - discr_si            -> R: discr_si()
  - overall_infectivity -> R: overall_infectivity()
  - estimate_R          -> R: estimate_R() with method = "parametric_si"
  - aggregate_inc       -> R: aggregate_inc()
  - estimate_R_agg      -> R: estimate_R_agg()  (the paper's EM algorithm)
  - simulate_incidence  -> R: projections::project(instantaneous_R = TRUE)
                           (Poisson branching process / discrete renewal process)

Numerical recipes (Cori et al. AJE 2013): with serial interval pmf g_s and a
Gamma(shape = a, scale = b) prior on R, the posterior over window [t1, t2] is
Gamma(shape = a + sum_t I_t, scale = 1 / (1/b + sum_t lambda_t)),
where lambda_t = sum_s g_s * I_{t-s} is the total infectivity. The paper's EM
extension reconstructs daily incidence inside each aggregation window from the
window's posterior mean R* converted to a growth rate r via the
Wallinga-Lipsitch link R = 1 / sum_s exp(-r s) g_s.
"""
import numpy as np
import pandas as pd
from scipy.special import gammainc
from scipy.stats import gamma

QUANTILES = [0.025, 0.05, 0.25, 0.5, 0.75, 0.95, 0.975]
QUANTILE_COLS = ["Quantile.0.025(R)", "Quantile.0.05(R)", "Quantile.0.25(R)",
                 "Median(R)", "Quantile.0.75(R)", "Quantile.0.95(R)",
                 "Quantile.0.975(R)"]


def discr_si(k, mu, sigma):
    """Discretised serial interval pmf (shifted Gamma, shift 1; Cori et al. 2013).
    k: int or array of days (>= 0). Matches R EpiEstim::discr_si exactly.
    Used for the R posterior (overall_infectivity), NOT for the R->r link."""
    k = np.asarray(k, dtype=float)
    if sigma < 0:
        raise ValueError("sigma must be >= 0")
    if mu <= 1:
        raise ValueError("mu must be > 1")
    if np.any(k < 0):
        raise ValueError("all k must be >= 0")
    a = ((mu - 1) / sigma) ** 2
    b = sigma ** 2 / (mu - 1)
    # pgamma(x, shape, scale) == gammainc(shape, x / scale)
    cdf = lambda x, shape: np.where(x > 0, gammainc(shape, np.where(x > 0, x, 0) / b), 0.0)
    res = (k * cdf(k, a)
           + (k - 2) * cdf(k - 2, a)
           - 2 * (k - 1) * cdf(k - 1, a)
           + a * b * (2 * cdf(k - 1, a + 1)
                      - cdf(k - 2, a + 1)
                      - cdf(k, a + 1)))
    return np.maximum(res, 0.0)


def gamma_mucv2shapescale(mu, sd):
    """epitrix::gamma_mucv2shapescale(mu, cv = sd/mu): shape = 1/cv^2,
    scale = mu * cv^2."""
    cv = sd / mu
    return 1.0 / cv ** 2, mu * cv ** 2


def si_distcrete_gamma(mu, sd, n=1001):
    """Serial interval used for the R->r link by the EpiEstim build that
    generated the paper's stored results: distcrete("gamma", interval = 1,
    shape, scale, w = 0.5), i.e. the centred discretisation
      d(k) = F(k + 0.5) - F(k - 0.5),  k = 0, ..., n-1
    of the UNshifted Gamma(shape = 1/cv^2, scale = mu*cv^2), with epitrix's
    truncation at day 1000 (n = 1001 masses). See the paper repo's
    vignette/functions.R (get_r_from_R). This differs from discr_si(): same
    mean/sd but different low-day masses (d(0) = F(0.5) > 0) and no 1e-6
    truncation, which changes the link R(r) noticeably for r > 0."""
    shape, scale = gamma_mucv2shapescale(mu, sd)
    k = np.arange(n)
    x0 = (k - 0.5) / scale
    x1 = (k + 0.5) / scale
    cdf = lambda x: np.where(x > 0, gammainc(shape, np.where(x > 0, x, 0)), 0.0)
    return cdf(x1) - cdf(x0)


def process_I(incid):
    """R: process_I(). Returns (local, imported, total). The first day's cases
    are moved to 'imported' (local[0] = 0)."""
    I = np.asarray(incid, dtype=float)
    local = I.copy()
    local[0] = 0.0
    imported = np.zeros_like(I)
    imported[0] = I[0]
    return local, imported


def overall_infectivity(incid, si_distr):
    """R: overall_infectivity(). lambda[t] = sum_s w_s * I_{t-s}
    (I = local + imported). 1-based day t: si_distr[1..t] * I[t..1], i.e.
    w_0..w_{t-1} times I_t..I_1. lambda[0] = NA (returned as NaN)."""
    I = np.asarray(incid, dtype=float)
    T = len(I)
    lam = np.full(T, np.nan)
    for t in range(1, T):
        # 0-based lam[t] = 1-based day t+1: w_0..w_t times I_t..I_0
        lam[t] = np.sum(si_distr[:t + 1] * I[t::-1])
    return lam


def estimate_R(incid, mean_si=None, std_si=None, si_distr=None,
               mean_prior=5.0, std_prior=5.0, t_start=None, t_end=None):
    """R: estimate_R(method = "parametric_si"), analytic Gamma posterior.
    Returns dict with 'R' (DataFrame) and 'si_distr'."""
    I = np.asarray(incid, dtype=float)
    T = len(I)
    local, imported = process_I(I)

    if si_distr is None:
        si_distr = discr_si(np.arange(0, T), mean_si, std_si)   # w_0 .. w_{T-1}
    lam = overall_infectivity(I, si_distr)
    final_mean_si = float(np.sum(si_distr * np.arange(len(si_distr))))

    if t_start is None:
        t_start = np.arange(2, T - 6 + 1)      # R: seq(2, T-6)
        t_end = np.arange(8, T + 1)            # R: seq(8, T)
    else:
        t_start = np.asarray(t_start, dtype=int)
        t_end = np.asarray(t_end, dtype=int)

    a_prior = (mean_prior / std_prior) ** 2
    b_prior = std_prior ** 2 / mean_prior

    # The EpiEstim build that generated the paper's stored results dropped
    # windows with t_end <= mean_si (raw mean, not the truncated
    # final_mean_si used by current EpiEstim). With the default windows
    # (t_start = seq(2, T-6), t_end = seq(8, T)) nothing is dropped for the
    # paper's SI parameters; the visible drops in the stored daily RDS files
    # come from the custom 2-day windows (t_start = 2..T-1,
    # t_end = t_start + 1) in main_analysis/*.R: flu t_end starts at 4,
    # covid at 7.
    thresh = mean_si if mean_si is not None else final_mean_si

    n = len(t_start)
    a_post = np.full(n, np.nan)
    b_post = np.full(n, np.nan)
    for i in range(n):
        if t_end[i] > thresh:
            a_post[i] = a_prior + local[t_start[i] - 1:t_end[i]].sum()
            b_post[i] = 1.0 / (1.0 / b_prior
                               + np.nansum(lam[t_start[i] - 1:t_end[i]]))
        # else: NA (window ends before mean SI)

    mean_post = a_post * b_post
    std_post = np.sqrt(a_post) * b_post

    # R drops non-estimable windows from the output (they do not appear at
    # all in the stored RDS); keep only estimable rows.
    keep = t_end > thresh
    t_start, t_end = t_start[keep], t_end[keep]
    a_post, b_post = a_post[keep], b_post[keep]
    mean_post, std_post = mean_post[keep], std_post[keep]

    R = pd.DataFrame({
        "t_start": t_start, "t_end": t_end,
        "Mean(R)": mean_post, "Std(R)": std_post,
    })
    for q, col in zip(QUANTILES, QUANTILE_COLS):
        R[col] = gamma.ppf(q, a_post, scale=b_post)

    return {"R": R, "si_distr": si_distr, "lambda": lam,
            "a_posterior": a_post, "b_posterior": b_post,
            "I": I, "I_local": local, "I_imported": imported}


def aggregate_inc(incid, dt=7):
    """R: aggregate_inc(). Sum daily incidence over windows of length dt."""
    I = np.asarray(incid, dtype=float)
    n = len(I)
    starts = np.arange(0, n - (dt - 1), dt)
    return np.array([I[s:s + dt].sum() for s in starts])


def _r2r0(r, si_distr):
    """Wallinga-Lipsitch link exactly as epitrix::r2R0 for a numeric w
    (continuous time, bin midpoints):
      t = [0, 0.5, 1.5, ..., n-0.5]   (w recycled with w[0])
      R(r) = r / sum_s -w_s * diff(exp(-r t)) / diff(t),   R(0) = 1
    Since w_0 = 0 this equals (r / (2 sinh(r/2))) / sum_s w_s exp(-r s) for
    the discr_si case. epitrix also has the guard: when exp(-r * t_last)
    overflows (== Inf in R), R0 is set to 0. For the distcrete SI (1001
    masses, t_last = 1000.5) this triggers for r < -0.709, which the grid
    search relies on; for the truncated discr_si it is unreachable."""
    r = np.asarray(r, dtype=float)
    out = np.empty_like(r, dtype=float)
    near0 = np.abs(r) < 1e-14
    out[near0] = 1.0
    if np.all(near0):
        return out
    t = np.concatenate([[0.0],
                        np.arange(0.5, len(si_distr) - 0.5 + 1e-12, 1.0)])
    rr = r[~near0]
    with np.errstate(over="ignore", invalid="ignore"):
        expmt = np.exp(-rr[:, None] * t)
        denom = -(si_distr * np.diff(expmt, axis=1) / np.diff(t))
        val = rr / denom.sum(axis=1)
        val = np.where(np.isinf(expmt[:, -1]), 0.0, val)
    out[~near0] = val
    return out


def _get_r_from_R(R_vals, mean_si, std_si, total_t, grid,
                  si_variant="distcrete"):
    """Growth rate for each R* via grid search over r (port of the R helper
    get_r_from_R inside estimate_R_agg).

    si_variant controls the serial interval used in the link R(r):
      - "distcrete" (default): distcrete("gamma", w = 0.5) centred
        discretisation, support 0..1000 (si_distcrete_gamma). This is what
        the EpiEstim build that generated the paper's stored results used
        (paper repo vignette/functions.R) and reproduces them to 1e-10.
      - "discr_si": discr_si(0..total_t) with the tail below 1e-6 dropped
        and renormalised — what current EpiEstim GitHub main does.
    The two give identical answers for R* < 1 (negative r) but differ for
    rising epidemics, where the link is most sensitive."""
    precision, gmin, gmax = grid["precision"], grid["min"], grid["max"]

    if si_variant == "distcrete":
        si = si_distcrete_gamma(mean_si, std_si)
        si = si / si.sum()                      # epitrix::r2R0 renormalises
    elif si_variant == "discr_si":
        si = discr_si(np.arange(0, total_t + 1), mean_si, std_si)
        keep = np.concatenate([[True], si[1:] >= 1e-6])
        si = si[keep]
        si = si / si.sum()
    else:
        raise ValueError("si_variant must be 'distcrete' or 'discr_si'")

    r_grid = np.arange(gmin, gmax + precision / 2, precision)
    R_grid = _r2r0(r_grid, si)
    idx = np.array([np.argmin(np.abs(R_grid - Rv)) for Rv in R_vals])

    # if any R* sits at an edge of the r grid, widen it (x5) and re-search;
    # this is the loop in the R code. For the paper's datasets it terminates
    # after at most one widening; the cap only guards pathological input.
    n_widen = 0
    while np.any(idx == 0) or np.any(idx == len(r_grid) - 1):
        if n_widen >= 20:
            raise RuntimeError("r grid widening did not terminate")
        mult = 5
        gmax = mult * gmax if gmax > 0 else -gmax
        gmin = mult * gmin if gmin < 0 else -gmin
        r_grid = np.arange(gmin, gmax + precision / 2, precision)
        R_grid = _r2r0(r_grid, si)
        idx = np.array([np.argmin(np.abs(R_grid - Rv)) for Rv in R_vals])
        n_widen += 1
    return r_grid[idx]


def estimate_R_agg(incid, dt=7, dt_out=7, iter=10, tol=1e-6,
                   recon_opt="naive", mean_si=None, std_si=None,
                   mean_prior=5.0, std_prior=5.0,
                   grid=None, si_variant="distcrete", verbose=False):
    """The paper's EM algorithm (port of R EpiEstim::estimate_R_agg, scalar dt).
    Returns dict with 'R' (final Rt estimates), 'I' (reconstructed daily
    incidence), 'I_local', 'I_imported', 'iterations' (per-iteration
    reconstructions), 'si_distr'.
    si_variant: which serial interval the R->r link uses ("distcrete" =
    the paper's stored results; "discr_si" = current EpiEstim main). The
    posterior always uses discr_si, as in EpiEstim."""
    if grid is None:
        grid = {"precision": 0.001, "min": -1.0, "max": 1.0}

    incid = np.asarray(incid, dtype=float)
    n_dt = len(incid)
    total_t = n_dt * dt

    # Fixed windows for R* (matched to aggregation windows; estimation starts
    # on the 1st day of the 2nd aggregation window).
    t_start = np.arange(dt + 1, total_t - (dt - 1) + 1, dt)
    t_end = t_start + (dt - 1)

    full_dt = np.repeat(dt, n_dt)
    sim_inc = np.full((total_t, iter), np.nan)

    # Naive disaggregation used to initialise the EM and to fill days that
    # cannot be reconstructed.
    dis = incid / dt
    dis_inc = np.repeat(dis, dt)

    min_t_start = None
    for i in range(iter):
        cur = dis_inc if i == 0 else sim_inc[:, i - 1]

        res_R = estimate_R(cur, mean_si=mean_si, std_si=std_si,
                           mean_prior=mean_prior, std_prior=std_prior,
                           t_start=t_start, t_end=t_end)
        mean_R = res_R["R"]["Mean(R)"].to_numpy()
        tstarts = res_R["R"]["t_start"].to_numpy()
        ok = ~np.isnan(mean_R)
        mean_R = mean_R[ok]
        tstarts = tstarts[ok]
        d0 = int(tstarts[0])                    # 1-based first estimable day
        min_t_start = d0
        # days to reconstruct (1-based d0..total_t)
        idx_reconstruct = np.arange(d0 - 1, total_t)

        idx_aggregation = np.repeat(np.arange(1, n_dt + 1), full_dt)
        first_win = int(idx_aggregation[d0 - 1])   # 1-based window of day d0

        if recon_opt == "naive":
            aggs = np.arange(first_win, n_dt + 1)          # 1-based windows
        elif recon_opt == "match":
            aggs = np.arange(first_win - 1, n_dt + 1)
            # also reconstruct the preceding window with the same growth rate
            add_idx = np.arange(d0 - dt - 1, d0 - 1)   # 0-based days d0-dt .. d0-1
            idx_reconstruct = np.concatenate([add_idx, idx_reconstruct])
        else:
            raise ValueError("recon_opt must be 'naive' or 'match'")

        mask = np.zeros(total_t, dtype=bool)
        mask[idx_reconstruct] = True
        incid_not_to_reconstruct = dis_inc[~mask]
        incid_to_reconstruct = incid[aggs - 1]

        gr = _get_r_from_R(mean_R, mean_si, std_si, total_t, grid,
                           si_variant=si_variant)
        if recon_opt == "match":
            gr = np.concatenate([[gr[0]], gr])

        dt_seq = full_dt[aggs - 1]
        k = np.empty(len(aggs))
        for w in range(len(aggs)):
            if dt_seq[w] > 1:
                d = np.sum(np.exp(gr[w] * np.arange(1, dt_seq[w])))
                k[w] = incid_to_reconstruct[w] / (np.exp(gr[w]) * (1 + d))
            else:
                k[w] = incid_to_reconstruct[w]
                gr[w] = 0.0

        est_inc = np.empty(int(dt_seq.sum()))
        pos = 0
        for w in range(len(aggs)):
            w_day = np.arange(1, dt_seq[w] + 1)
            est_inc[pos:pos + dt_seq[w]] = k[w] * np.exp(gr[w] * w_day)
            pos += dt_seq[w]

        sim_inc[:, i] = np.concatenate([incid_not_to_reconstruct, est_inc])

        if i == iter - 1:
            if np.any(np.abs(sim_inc[:, i] - sim_inc[:, i - 1]) > tol):
                if verbose:
                    print("Reconstructed incidence has not converged within "
                          "the set tolerance; rerun with more iterations.")
            out_start = np.arange(min_t_start, total_t - (dt_out - 1) + 1)
            out_end = out_start + (dt_out - 1)
            R_out = estimate_R(sim_inc[:, i], mean_si=mean_si, std_si=std_si,
                               mean_prior=mean_prior, std_prior=std_prior,
                               t_start=out_start, t_end=out_end)

    return {"R": R_out["R"], "I": sim_inc[:, iter - 1],
            "I_local": R_out["I_local"], "I_imported": R_out["I_imported"],
            "iterations": sim_inc, "si_distr": R_out["si_distr"],
            "mean_si": mean_si, "std_si": std_si}


def simulate_incidence(Rt, si_one, n_days=70, n_sim=100, initial=None,
                       seed=None):
    """Poisson branching process (renewal process), matching
    R projections::project(instantaneous_R = TRUE):
      I_t ~ Pois(R_t * sum_s I_{t-s} g_s), g = si_one (s = 1, 2, ...).
    Rt: scalar or array broadcast to n_days; initial: incidence before t=0
    (its length must be >= len(si_one)). Returns (n_sim, n_days)."""
    rng = np.random.default_rng(seed)
    si_one = np.asarray(si_one, dtype=float)
    if initial is None:
        initial = np.repeat(10.0, 7)   # as in simulate_incidence.R
    initial = np.asarray(initial, dtype=float)
    history = np.concatenate([initial, np.zeros(n_days)])
    Rt = np.broadcast_to(np.asarray(Rt, dtype=float), (n_days,))

    sims = np.empty((n_sim, n_days))
    L = len(si_one)
    for s in range(n_sim):
        hist = history.copy()
        for t in range(n_days):
            off = len(initial) + t
            lo = max(0, off - L)
            past = hist[lo:off]                       # oldest .. most recent
            m = len(past)
            mean = Rt[t] * np.sum(past[::-1] * si_one[:m])   # lag 1..m
            hist[off] = rng.poisson(mean)
        sims[s] = hist[len(initial):]
    return sims
