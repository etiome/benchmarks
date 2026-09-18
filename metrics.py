"""
metrics.py — per-method evaluation metrics beyond GSEA enrichment.

Metrics
-------
iep_correlation
    Pearson r between a method's pseudotime and the cell-level iEP score.
    Only meaningful for methods that produce a continuous pseudotime.
    Returns NaN for statistical methods (ANOVA, Wilcoxon, DESeq2).

concordance_index
    Fraction of (sample_i, sample_j) pairs where the ordering of mean
    pseudotime agrees with the ordering of ordinal_condition.
    Range [0, 1]; 0.5 = random, 1.0 = perfect.

transcriptional_continuity
    Weighted lag-1 autocorrelation of mean pairwise cell distances along
    the pseudotime axis (binned). Higher = smoother trajectory.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# IEP correlation
# ---------------------------------------------------------------------------

def iep_correlation(pseudotime: np.ndarray, obs: pd.DataFrame) -> float:
    """Pearson r between pseudotime and obs['iep_score']. Returns NaN if unavailable."""
    if "iep_score" not in obs.columns:
        return float("nan")
    from scipy.stats import pearsonr
    iep = obs["iep_score"].values.astype(float)
    pt  = np.asarray(pseudotime, dtype=float)
    valid = np.isfinite(pt) & np.isfinite(iep)
    if valid.sum() < 2:
        return float("nan")
    r, _ = pearsonr(pt[valid], iep[valid])
    return float(r)


# ---------------------------------------------------------------------------
# Concordance index
# ---------------------------------------------------------------------------

def concordance_index(pseudotime: np.ndarray, obs: pd.DataFrame) -> float:
    """
    Sample-level concordance: fraction of sample pairs where mean pseudotime
    order agrees with ordinal_condition order.
    """
    if "sample" not in obs.columns or "ordinal_condition" not in obs.columns:
        return float("nan")

    pt = np.asarray(pseudotime, dtype=float)
    valid = np.isfinite(pt)
    if valid.sum() == 0:
        return float("nan")

    tmp = obs[["sample", "ordinal_condition"]].copy()
    tmp["pt"] = pt
    tmp = tmp[valid]

    # Sample-level mean pseudotime and condition
    sample_pt   = tmp.groupby("sample")["pt"].mean()
    sample_cond = tmp.groupby("sample")["ordinal_condition"].first()

    samples = sample_pt.index
    n = len(samples)
    if n < 2:
        return float("nan")

    concordant = discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            pt_diff   = sample_pt.iloc[i]   - sample_pt.iloc[j]
            cond_diff = sample_cond.iloc[i] - sample_cond.iloc[j]
            if cond_diff == 0:
                continue  # tied condition — ignore pair
            if (pt_diff > 0) == (cond_diff > 0):
                concordant += 1
            else:
                discordant += 1

    total = concordant + discordant
    return float(concordant / total) if total > 0 else float("nan")


# ---------------------------------------------------------------------------
# Transcriptional continuity
# ---------------------------------------------------------------------------

def transcriptional_continuity(
    pseudotime: np.ndarray,
    adata,
    n_bins: int = 50,
    sample_size: int = 500,
    lag: int = 1,
    seed: int = 42,
) -> float:
    """
    Weighted lag-1 autocorrelation of inter-bin cell distances along pseudotime.

    Bins cells by pseudotime, computes mean pairwise Euclidean distance between
    each pair of bins (in PCA space), then returns the autocorrelation of that
    distance series weighted by bin occupancy.

    Uses PCA on gene expression (30 components) if no precomputed embedding is
    available in adata.obsm.
    """
    from scipy.spatial.distance import cdist

    pt = np.asarray(pseudotime, dtype=float)
    valid = np.isfinite(pt)
    if valid.sum() <= 10:
        return float("nan")

    try:
        adata_v = adata[valid]
        obs_names = adata_v.obs_names

        # Feature matrix: prefer X_trajectory obsm, else PCA on X
        if "X_trajectory" in adata_v.obsm:
            features = pd.DataFrame(adata_v.obsm["X_trajectory"], index=obs_names)
        else:
            import scanpy as sc
            tmp = adata_v.copy()
            sc.pp.pca(tmp, n_comps=min(30, adata_v.n_vars - 1))
            features = pd.DataFrame(tmp.obsm["X_pca"], index=obs_names)

        pt_valid = pt[valid]
        emb = pd.DataFrame({"embedding": pt_valid}, index=obs_names)

        bins = np.linspace(pt_valid.min(), pt_valid.max(), n_bins)
        emb["bin"] = np.digitize(pt_valid, bins)
        bin_avg = emb.groupby("bin")["embedding"].mean()
        bin_ids = sorted(bin_avg.index.tolist())

        bin_proportions = (
            emb["bin"].value_counts(normalize=True)
            .reindex(bin_ids)
            .fillna(0.0)
        )

        rng = np.random.default_rng(seed)
        bin_samples: dict[int, np.ndarray | None] = {}
        for bid in bin_ids:
            idx = emb.index[emb["bin"] == bid].tolist()
            if not idx:
                bin_samples[bid] = None
                continue
            X = features.loc[idx].values
            if len(X) > sample_size:
                chosen = rng.choice(len(X), sample_size, replace=False)
                X = X[chosen]
            bin_samples[bid] = X

        nb = len(bin_ids)
        distances = np.full((nb, nb), np.nan)
        for ii, bi in enumerate(bin_ids):
            xi = bin_samples[bi]
            if xi is None:
                continue
            for jj in range(ii, nb):
                bj = bin_ids[jj]
                xj = bin_samples[bj]
                if xj is None:
                    continue
                d = cdist(xi, xj, metric="euclidean").mean()
                distances[ii, jj] = d
                distances[jj, ii] = d

        autocorr = []
        for ii in range(nb):
            row = np.concatenate([distances[ii, :ii], distances[ii, ii + 1:]])
            row = row[np.isfinite(row)]
            ac = pd.Series(row).autocorr(lag=lag) if len(row) >= lag + 1 else np.nan
            autocorr.append(ac)

        weights = bin_proportions.values
        ac_arr  = np.array(autocorr)
        score   = float(np.nansum(ac_arr * weights))
        return score

    except Exception as exc:
        print(f"  Warning: transcriptional_continuity failed: {exc}")
        return float("nan")
