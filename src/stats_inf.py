"""
Statistical inference utilities for graduation data.
Functions:
 - chi2_cramersv
 - cohens_d
 - mannwhitney_r
 - run_hypothesis_tests (returns JSON results)
"""

import logging
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from pathlib import Path
from src.data_utils import load_participants

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(exist_ok=True)
FIG_DIR = REPORT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

def chi2_cramersv(x, y):
    ct = pd.crosstab(x, y)
    chi2, p, dof, _ = stats.chi2_contingency(ct)
    n = ct.sum().sum()
    r, k = ct.shape
    cramers_v = np.sqrt((chi2 / (n * (min(r, k) - 1)))) if min(r, k) > 1 else np.nan
    return {"chi2": chi2, "p": p, "dof": dof, "cramers_v": cramers_v}

def cohens_d(a, b):
    a, b = np.asarray(a), np.asarray(b)
    n1, n2 = len(a), len(b)
    s1, s2 = np.var(a, ddof=1), np.var(b, ddof=1)
    s = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2)) if (n1 + n2 - 2) > 0 else np.nan
    return (np.mean(a) - np.mean(b)) / s if s else np.nan

def mannwhitney_r(a, b):
    u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    n1, n2 = len(a), len(b)
    mean_u, std_u = n1 * n2 / 2, np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
    z = (u - mean_u) / std_u if std_u > 0 else 0
    return {"u": int(u), "p": p, "r": z / np.sqrt(n1 + n2)}

#Run chi-square, t-tests, Mann-Whitney, correlations, and FDR correction
def run_hypothesis_tests(df):
    logging.info("Running hypothesis tests on %d rows", len(df))
    results, md_lines = {}, []

    # 1. track vs graduation
    res1 = chi2_cramersv(df["track_name"].fillna("MISSING"), df["graduation_status"])
    results["track_vs_grad"] = res1
    md_lines.append(f"H1: Track vs Graduation → χ²={res1['chi2']:.2f}, p={res1['p']:.4f}, V={res1['cramers_v']:.2f}")

    # 2. gender vs graduation
    res2 = chi2_cramersv(df["gender"].fillna("MISSING"), df["graduation_status"])
    results["gender_vs_grad"] = res2
    md_lines.append(f"H2: Gender vs Graduation → χ²={res2['chi2']:.2f}, p={res2['p']:.4f}, V={res2['cramers_v']:.2f}")

    # 3. total_score grads vs non
    grad_scores = df[df["graduation_status"]]["total_score"].dropna()
    nongrad_scores = df[~df["graduation_status"]]["total_score"].dropna()
    ttest = stats.ttest_ind(grad_scores, nongrad_scores, equal_var=False)
    d = cohens_d(grad_scores, nongrad_scores)
    mw = mannwhitney_r(grad_scores, nongrad_scores)
    results["total_score"] = {"t": ttest.statistic, "p": ttest.pvalue, "cohen_d": d, "mw": mw}
    md_lines.append(f"H3: Total Score vs Graduation → t={ttest.statistic:.2f}, p={ttest.pvalue:.4f}, d={d:.2f}")

    # 4. hours per week
    grad_hours = df[df["graduation_status"]]["hours_per_week"].dropna()
    nongrad_hours = df[~df["graduation_status"]]["hours_per_week"].dropna()
    ttest2 = stats.ttest_ind(grad_hours, nongrad_hours, equal_var=False)
    d2 = cohens_d(grad_hours, nongrad_hours)
    mw2 = mannwhitney_r(grad_hours, nongrad_hours)
    results["hours"] = {"t": ttest2.statistic, "p": ttest2.pvalue, "cohen_d": d2, "mw": mw2}
    md_lines.append(f"H4: Hours vs Graduation → t={ttest2.statistic:.2f}, p={ttest2.pvalue:.4f}, d={d2:.2f}")

    # correlation
    corr_cols = [c for c in ["total_score", "hours_per_week", "years_experience", "skill_level"] if c in df.columns]
    corr = df[corr_cols].corr(method="spearman") if corr_cols else pd.DataFrame()
    results["spearman_corr"] = corr.to_dict()

    # multiple testing correction
    pvals = [res1["p"], res2["p"], ttest.pvalue, ttest2.pvalue]
    reject, pv_corr, _, _ = multipletests(pvals, alpha=0.05, method="fdr_bh")
    results["multiple_testing"] = {"raw": pvals, "adj": pv_corr.tolist(), "reject": reject.tolist()}

    return results

# For standalone runs
if __name__ == "__main__":
    df = load_participants()
    res = run_hypothesis_tests(df)
    print(res)
