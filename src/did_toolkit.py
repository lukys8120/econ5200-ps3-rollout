"""Difference-in-differences toolkit (ECON 5200 PS3), extending Lab 11's did_utils.py."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

WEBB_WEIGHTS = np.array([-np.sqrt(1.5), -1.0, -np.sqrt(0.5), np.sqrt(0.5), 1.0, np.sqrt(1.5)])


def _never_treated(df: pd.DataFrame, time: str, cohort: str) -> pd.Series:
    """Flag never-treated units (cohort is inf or after the last period)."""
    adoption = df[cohort].astype(float)
    return ~np.isfinite(adoption) | (adoption > df[time].max())


def two_by_two(df: pd.DataFrame, outcome: str, time: str, cohort: str,
               treated_cohort: float, control_cohort: float,
               pre: Sequence[int], post: Sequence[int]) -> float:
    """Simple 2x2 diff-in-diff of cohort means between two cohorts."""
    def cohort_mean(g: float, periods: Sequence[int]) -> float:
        in_cell = (df[cohort] == g) & df[time].isin(periods)
        return df.loc[in_cell, outcome].mean()

    return ((cohort_mean(treated_cohort, post) - cohort_mean(treated_cohort, pre))
            - (cohort_mean(control_cohort, post) - cohort_mean(control_cohort, pre)))


def event_study(df: pd.DataFrame, outcome: str, unit: str, time: str, cohort: str,
                ref_period: int = -1) -> pd.Series:
    """Event-study coefficients by event time, relative to `ref_period`."""
    data = df.copy()
    never = _never_treated(data, time, cohort)
    data["_event_time"] = np.where(never, np.nan, data[time] - data[cohort])
    event_times = sorted(int(e) for e in data["_event_time"].dropna().unique())

    event_time_column = {}
    for e in event_times:
        if e == ref_period:
            continue
        event_time_column[e] = f"pre{-e}" if e < 0 else f"post{e}"
        data[event_time_column[e]] = (data["_event_time"] == e).astype(int)

    formula = (f"{outcome} ~ " + " + ".join(event_time_column.values())
               + f" + C({unit}) + C({time})")
    fit = smf.ols(formula, data=data).fit()
    return pd.Series({e: 0.0 if e == ref_period else fit.params[event_time_column[e]]
                      for e in event_times}, name="estimate")


def clean_att(df: pd.DataFrame, outcome: str, time: str, cohort: str) -> float:
    """Overall ATT using only not-yet- and never-treated controls, weighted by cell size."""
    never = _never_treated(df, time, cohort)
    adoption = df[cohort].where(~never, np.inf)
    cohorts = sorted(adoption[~never].unique())
    periods = sorted(df[time].unique())

    def mean_at(rows: pd.Series, t: int) -> float:
        return df.loc[rows & (df[time] == t), outcome].mean()

    cell_att, cell_size = [], []
    for g in cohorts:
        in_cohort = adoption == g
        for t in (p for p in periods if p >= g):
            controls = adoption > t
            if not controls.any():
                continue
            cell_att.append((mean_at(in_cohort, t) - mean_at(in_cohort, g - 1))
                            - (mean_at(controls, t) - mean_at(controls, g - 1)))
            cell_size.append((in_cohort & (df[time] == t)).sum())
    return float(np.average(cell_att, weights=cell_size))




def clustered_fit(df: pd.DataFrame, formula: str, term: str,
                  cluster: str | None = None) -> pd.Series:
    """One coefficient with default SEs, or SEs clustered on `cluster`."""
    model = smf.ols(formula, data=df)
    if cluster is None:
        fit = model.fit()
        n_clusters = len(df)
    else:
        fit = model.fit(cov_type="cluster", cov_kwds={"groups": df[cluster]}, use_t=True)
        n_clusters = df[cluster].nunique()
    return pd.Series({"estimate": fit.params[term], "se": fit.bse[term],
                      "t": fit.tvalues[term], "p": fit.pvalues[term],
                      "clusters": n_clusters})



def wild_cluster_bootstrap(df: pd.DataFrame, outcome: str, treatment: str, unit: str,
                           time: str, cluster: str, n_boot: int = 9999,
                           seed: int = 42) -> pd.Series:
    """Wild cluster bootstrap p-value for the TWFE coefficient, null imposed, Webb weights."""
    y = df[outcome].to_numpy(float)
    fixed_effects = pd.get_dummies(df[[unit, time]].astype(str), drop_first=True).to_numpy(float)
    X_restricted = np.column_stack([np.ones(len(df)), fixed_effects])
    X = np.column_stack([df[treatment].to_numpy(float), X_restricted])

    codes, _ = pd.factorize(df[cluster])
    n_clusters = codes.max() + 1
    n, k = X.shape
    cluster_indicator = np.zeros((n_clusters, n))
    cluster_indicator[codes, np.arange(n)] = 1.0

    projection = np.linalg.pinv(X.T @ X) @ X.T
    small_sample = n_clusters / (n_clusters - 1) * (n - 1) / (n - k)

    def t_stats(Y: np.ndarray) -> np.ndarray:
        beta = projection @ Y
        resid = Y - X @ beta
        scores = cluster_indicator @ (projection[0][:, None] * resid)
        return beta[0] / np.sqrt(small_sample * (scores ** 2).sum(axis=0))

    t_observed = t_stats(y[:, None])[0]

    beta_restricted, *_ = np.linalg.lstsq(X_restricted, y, rcond=None)
    fitted_restricted = X_restricted @ beta_restricted
    resid_restricted = y - fitted_restricted

    rng = np.random.default_rng(seed)
    weights = rng.choice(WEBB_WEIGHTS, size=(n_clusters, n_boot))
    Y_boot = fitted_restricted[:, None] + resid_restricted[:, None] * weights[codes]
    t_boot = t_stats(Y_boot)

    return pd.Series({"t": t_observed,
                      "p": float(np.mean(np.abs(t_boot) >= abs(t_observed))),
                      "clusters": n_clusters, "draws": n_boot})


def _toy_panel(noise_sd: float = 0.0, effect_scale: float = 1.0, seed: int = 0) -> pd.DataFrame:
    """Small staggered test panel with known effects."""
    rng = np.random.default_rng(seed)
    adoption = {0: 3, 1: 3, 2: 5, 3: 5, 4: np.inf, 5: np.inf}
    rows = []
    for unit in range(12):
        cluster = unit // 2
        g = adoption[cluster]
        for t in range(1, 9):
            treated = int(t >= g)
            effect = effect_scale * (1.0 + (t - g)) if treated else 0.0
            rows.append({"unit": unit, "cluster": cluster, "time": t, "cohort": g,
                         "treated": treated, "true_effect": effect,
                         "y": unit + 0.5 * t + effect + rng.normal(0, noise_sd)})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    clean = _toy_panel()
    noisy = _toy_panel(noise_sd=1.0, seed=1)
    strong = _toy_panel(noise_sd=0.5, effect_scale=10.0, seed=2)
    periods = np.arange(1, 9)

    assert np.isclose(two_by_two(clean, "y", "time", "cohort", 3, np.inf,
                                 periods[periods < 3], periods[periods >= 3]),
                      clean.loc[(clean.cohort == 3) & (clean.time >= 3), "true_effect"].mean())

    path = event_study(clean, "y", "unit", "time", "cohort")
    truth = pd.Series({e: (1.0 + e) if e >= 0 else 0.0 for e in path.index})
    assert np.allclose(path, truth) and path[-1] == 0.0

    assert np.isclose(clean_att(clean, "y", "time", "cohort"),
                      clean.loc[clean.treated == 1, "true_effect"].mean())

    formula = "y ~ treated + C(unit) + C(time)"
    fits = [clustered_fit(noisy, formula, "treated", c) for c in (None, "unit", "cluster")]
    assert np.allclose([f["estimate"] for f in fits], fits[0]["estimate"])
    assert all(f["se"] > 0 for f in fits)

    boot = wild_cluster_bootstrap(noisy, "y", "treated", "unit", "time", "cluster", n_boot=999)
    assert np.isclose(boot["t"], fits[2]["t"])
    assert 0.0 <= boot["p"] <= 1.0
    assert boot["p"] == wild_cluster_bootstrap(noisy, "y", "treated", "unit", "time",
                                               "cluster", n_boot=999)["p"]
    assert wild_cluster_bootstrap(strong, "y", "treated", "unit", "time", "cluster",
                                  n_boot=999)["p"] < 0.05

    print("All did_toolkit checks passed.")
