"""
Methods for trajectory reconstruction and gene ranking.

Each method receives a ``BenchmarkData`` object (from data_loader.py) and
returns a gene-ranking DataFrame with columns ``['activity', 'pvalue']``,
sorted descending by activity.
"""

from __future__ import annotations

import argparse
import json
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class TrajectoryMethod(ABC):
    """Base class for all trajectory / gene-ranking methods."""

    def __init__(self, data, input_type: str = "all_genes", **kwargs):
        """
        Parameters
        ----------
        data : BenchmarkData
            Loaded benchmark inputs (from data_loader.py).
        input_type : str
            One of 'all_genes' | 'harmony'.
            Controls what cell representation is used for trajectory inference.
        """
        self.data = data
        self.input_type = input_type
        self.adata = data.adata

    # ------------------------------------------------------------------
    # Input preparation
    # ------------------------------------------------------------------

    def _get_input(self) -> np.ndarray:
        """Return the cell × feature matrix for trajectory inference."""
        if self.input_type == "all_genes":
            X = self.adata.X
            return X.toarray() if hasattr(X, "toarray") else np.asarray(X)

        if self.input_type == "harmony":
            import scanpy as sc
            try:
                import harmonypy as hm
            except ImportError:
                raise ImportError("harmonypy not installed: pip install harmonypy")
            adata = self.adata.copy()
            sc.pp.pca(adata, n_comps=30, random_state=0)
            result = hm.run_harmony(adata.obsm["X_pca"].copy(), adata.obs, "sample",
                                     random_state=0, ncores=1)
            return result.Z_corr

        raise ValueError(
            f"Unknown input_type: {self.input_type!r}. Use 'all_genes' or 'harmony'."
        )

    # ------------------------------------------------------------------
    # Post-processing — restrict to genes present in the ranking table
    # ------------------------------------------------------------------

    def _align_to_valid_genes(self, gene_ranks: pd.DataFrame) -> pd.DataFrame:
        """Reindex to only genes present with a valid SNR in the ranking CSV."""
        valid = self.data.gene_ranking.index[self.data.gene_ranking["SNR"].notna()]
        if set(gene_ranks.index) != set(valid):
            gene_ranks = gene_ranks.reindex(valid)
            gene_ranks["activity"] = gene_ranks["activity"].fillna(0.0)
            gene_ranks["pvalue"] = gene_ranks["pvalue"].fillna(1.0)
        return gene_ranks

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def get_gene_ranks(self) -> pd.DataFrame:
        """Return DataFrame with columns ['activity', 'pvalue'] indexed by gene."""


# ---------------------------------------------------------------------------
# Helpers shared across pseudotime methods
# ---------------------------------------------------------------------------

def _pseudotime_gene_ranks(adata, pseudotime: np.ndarray) -> pd.DataFrame:
    """Spearman-correlate every gene with pseudotime; return ranked DataFrame."""
    X = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
    activities, pvalues = [], []
    for i in range(adata.n_vars):
        corr, pval = spearmanr(X[:, i], pseudotime)
        activities.append(abs(corr))
        pvalues.append(pval)
    return pd.DataFrame(
        {"activity": activities, "pvalue": pvalues}, index=adata.var_names
    ).sort_values("activity", ascending=False)


def _build_knn(adata, X: np.ndarray, input_type: str):
    """Compute a scanpy neighbour graph from X (harmony embedding or PCA of all_genes)."""
    import scanpy as sc
    if input_type == "harmony":
        # X is already a low-dim harmony embedding — safe to keep as an obsm key.
        adata.obsm["X_traj"] = X
        sc.pp.neighbors(adata, use_rep="X_traj", n_neighbors=15, random_state=0)
    else:
        # X here is the full dense gene matrix — do not stash it in obsm,
        # it would duplicate adata.X (~10GB) in the saved output file.
        sc.pp.pca(adata, n_comps=30, random_state=0)
        sc.pp.neighbors(adata, n_neighbors=15, random_state=0)


def _set_root(adata):
    """Set adata.uns['iroot'] to the first cell of the lowest ordinal condition."""
    cond_min = adata.obs["ordinal_condition"].min()
    root = adata.obs[adata.obs["ordinal_condition"] == cond_min].index[0]
    adata.uns["iroot"] = int(np.where(adata.obs_names == root)[0][0])


# ---------------------------------------------------------------------------
# PAGA, DPT, Palantir, TBR
# ---------------------------------------------------------------------------

class PAGAMethod(TrajectoryMethod):
    """PAGA graph + diffusion pseudotime → gene–pseudotime correlation."""

    def get_gene_ranks(self) -> pd.DataFrame:
        import scanpy as sc
        print("  Computing PAGA trajectory…")
        adata = self.adata.copy()
        _build_knn(adata, self._get_input(), self.input_type)
        sc.tl.leiden(adata, resolution=0.5)
        sc.tl.paga(adata, groups="leiden")
        # Root at most common cluster — matches original implementation
        root_cl = adata.obs["leiden"].value_counts().index[0]
        root_cell = adata.obs[adata.obs["leiden"] == root_cl].index[0]
        adata.uns["iroot"] = int(np.where(adata.obs_names == root_cell)[0][0])
        sc.tl.diffmap(adata)
        sc.tl.dpt(adata)
        pseudotime = adata.obs["dpt_pseudotime"].values
        self.pseudotime = pseudotime
        self.adata = adata
        print("  Correlating genes with pseudotime…")
        return self._align_to_valid_genes(_pseudotime_gene_ranks(adata, pseudotime))


class DPTMethod(TrajectoryMethod):
    """Diffusion pseudotime → gene–pseudotime correlation."""

    def get_gene_ranks(self) -> pd.DataFrame:
        import scanpy as sc
        print("  Computing DPT…")
        adata = self.adata.copy()
        _build_knn(adata, self._get_input(), self.input_type)
        sc.tl.diffmap(adata)
        _set_root(adata)
        sc.tl.dpt(adata)
        pseudotime = adata.obs["dpt_pseudotime"].values
        self.pseudotime = pseudotime
        self.adata = adata
        print("  Correlating genes with pseudotime…")
        return self._align_to_valid_genes(_pseudotime_gene_ranks(adata, pseudotime))


class PalantirMethod(TrajectoryMethod):
    """Palantir pseudotime → gene–pseudotime correlation."""

    def get_gene_ranks(self) -> pd.DataFrame:
        try:
            import palantir
        except ImportError:
            raise ImportError("palantir not installed: pip install palantir")
        import scanpy as sc
        print("  Computing Palantir pseudotime…")
        adata = self.adata.copy()
        X = self._get_input()
        if self.input_type == "harmony":
            adata.obsm["X_pca"] = X
        else:
            sc.pp.pca(adata, n_comps=min(300, X.shape[1] - 1), random_state=0)
        palantir.utils.run_diffusion_maps(adata, n_components=10, knn=50)
        palantir.utils.determine_multiscale_space(adata)
        cond_min = adata.obs["ordinal_condition"].min()
        start = adata.obs[adata.obs["ordinal_condition"] == cond_min].index[0]
        pr_res = palantir.core.run_palantir(adata, start, num_waypoints=300, knn=50)
        pseudotime = pr_res.pseudotime.values
        self.pseudotime = pseudotime
        self.adata = adata
        print("  Correlating genes with pseudotime…")
        return self._align_to_valid_genes(_pseudotime_gene_ranks(adata, pseudotime))


class TBRMethod(TrajectoryMethod):
    """Temporal biodynamics"""

    def get_gene_ranks(self) -> pd.DataFrame:
        self.pseudotime = self.data.embedding.reindex(self.adata.obs_names).values
        ranks = self.data.gene_ranking
        return pd.DataFrame(
            {"activity": ranks["SNR"], "pvalue": np.nan}
        ).sort_values("activity", ascending=False)



# ---------------------------------------------------------------------------
# Statistical baselines: ANOVA, Wilcoxon
# ---------------------------------------------------------------------------

class ANOVAMethod(TrajectoryMethod):
    """One-way ANOVA F-statistic across ordinal conditions."""

    def get_gene_ranks(self) -> pd.DataFrame:
        from sklearn.feature_selection import f_classif
        from tqdm import tqdm

        print("  Computing ANOVA F-statistics…")
        X = self.adata.X
        y = self.adata.obs["ordinal_condition"].values

        n_genes = self.adata.n_vars
        chunk_size = 1_000
        f_stats = np.empty(n_genes)
        pvalues = np.empty(n_genes)
        for start in tqdm(range(0, n_genes, chunk_size), desc="  ANOVA gene chunks"):
            stop = min(start + chunk_size, n_genes)
            chunk = X[:, start:stop]
            chunk = chunk.toarray() if hasattr(chunk, "toarray") else np.asarray(chunk)
            f_stats[start:stop], pvalues[start:stop] = f_classif(chunk, y)

        return self._align_to_valid_genes(
            pd.DataFrame({"activity": f_stats, "pvalue": pvalues}, index=self.adata.var_names)
            .sort_values("activity", ascending=False)
        )


class WilcoxonMethod(TrajectoryMethod):
    """Wilcoxon rank-sum test (each stage vs stage 0) via scanpy."""

    def get_gene_ranks(self) -> pd.DataFrame:
        import scanpy as sc
        print("  Computing Wilcoxon tests…")
        adata = self.adata.copy()
        adata.obs["cond_str"] = adata.obs["ordinal_condition"].astype(str)
        sc.tl.rank_genes_groups(adata, groupby="cond_str", reference="0", method="wilcoxon")
        result = adata.uns["rank_genes_groups"]
        groups = result["names"].dtype.names
        scores: dict[str, list] = {g: [] for g in adata.var_names}
        for grp in groups:
            if grp == "0":
                continue
            for gene, score in zip(result["names"][grp], result["scores"][grp]):
                scores[gene].append(abs(score))
        activities = {g: (np.mean(v) if v else 0.0) for g, v in scores.items()}
        return self._align_to_valid_genes(
            pd.DataFrame({"activity": list(activities.values()), "pvalue": np.nan},
                         index=list(activities.keys()))
            .sort_values("activity", ascending=False)
        )


# ---------------------------------------------------------------------------
# HiDDEN
# ---------------------------------------------------------------------------

class HiDDENMethod(TrajectoryMethod):
    """
    HiDDEN label refinement.

    Trains a logistic regression on control vs final-stage cells, projects all
    cells onto a perturbation-effect axis, binarises with K-means, then ranks
    genes by Wilcoxon DE between the two groups.
    """

    def get_gene_ranks(self) -> pd.DataFrame:
        from scipy.stats import ranksums
        from sklearn.cluster import KMeans
        from sklearn.decomposition import PCA
        from sklearn.linear_model import LogisticRegression

        print("  Running HiDDEN…")
        adata = self.adata.copy()
        conditions = sorted(adata.obs["ordinal_condition"].unique())
        ctrl, final = conditions[0], conditions[-1]
        X_raw = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
        X_input = self._get_input()

        mask = adata.obs["ordinal_condition"].isin([ctrl, final]).values
        # Fit PCA on ctrl+final only, then project all cells — matches original
        if self.input_type in ("features", "all_genes"):
            pca = PCA(n_components=30, random_state=0)
            pca.fit_transform(X_input[mask])   # fit on binary subset
            X_all = pca.transform(X_input)     # project all cells
            X_tr = X_all[mask]
        else:
            X_tr = X_input[mask]
            X_all = X_input
        y_tr = (adata.obs["ordinal_condition"].values[mask] == final).astype(int)

        model = LogisticRegression(penalty=None, max_iter=1000, random_state=0)
        model.fit(X_all[mask], y_tr)
        p_hat = model.predict_proba(X_all)[:, 1]

        km = KMeans(n_clusters=2, n_init="auto", random_state=0).fit(p_hat.reshape(-1, 1))
        labels = km.labels_
        if p_hat[labels == 0].mean() > p_hat[labels == 1].mean():
            labels = 1 - labels

        ctrl_mask = adata.obs["ordinal_condition"].values == ctrl
        affected = labels == 1
        print(f"    {ctrl_mask.sum()} control vs {affected.sum()} affected cells")

        activities, pvalues = [], []
        for i in range(adata.n_vars):
            stat, pval = ranksums(X_raw[ctrl_mask, i], X_raw[affected, i])
            activities.append(abs(stat))
            pvalues.append(pval)

        self.pseudotime = p_hat
        return self._align_to_valid_genes(
            pd.DataFrame({"activity": activities, "pvalue": pvalues}, index=adata.var_names)
            .sort_values("activity", ascending=False)
        )


# ---------------------------------------------------------------------------
# MELD
# ---------------------------------------------------------------------------

class MELDMethod(TrajectoryMethod):
    """
    MELD manifold enhancement.

    Computes per-cell condition likelihoods, then ranks genes by DE between
    cells enriched vs depleted in the final condition (VFC or quantile fallback).
    """

    def get_gene_ranks(self) -> pd.DataFrame:
        try:
            import meld
        except ImportError:
            raise ImportError("meld not installed: pip install meld")
        from scipy.stats import ranksums

        print("  Running MELD…")
        adata = self.adata.copy()
        conditions = sorted(adata.obs["ordinal_condition"].unique())
        X = self._get_input()
        X_raw = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)

        densities = meld.MELD(random_state=0).fit_transform(X, adata.obs["ordinal_condition"].values)
        likelihoods = meld.utils.normalize_densities(densities)
        max_lh = likelihoods.iloc[:, -1].values

        extreme = adata.obs["ordinal_condition"].isin([conditions[0], conditions[-1]]).values
        lh_ext = max_lh[extreme]

        # VFC's compute_fourier_basis() does a dense eigendecomposition of the
        # full NxN graph Laplacian (no sparse/truncated path in pygsp), so its
        # memory scales as O(N^2) with a much larger LAPACK workspace on top.
        # ~15k cells is enough to OOM-kill a 30GiB host. Cap the graph size by
        # subsampling "extreme" cells so VFC still runs on large datasets.
        VFC_MAX_CELLS = 5_000
        extreme_idx = np.flatnonzero(extreme)
        if extreme_idx.size > VFC_MAX_CELLS:
            rng = np.random.default_rng(0)
            extreme_idx = rng.choice(extreme_idx, size=VFC_MAX_CELLS, replace=False)
            extreme_idx.sort()
        lh_ext = max_lh[extreme_idx]

        import graphtools as gt
        from meld import VertexFrequencyCluster
        G = gt.Graph(X[extreme_idx], use_pygsp=True, random_state=0)
        G.compute_fourier_basis()
        vfc = VertexFrequencyCluster(n_clusters=10, random_state=0)
        vfc.fit_transform(G, adata.obs["ordinal_condition"].values[extreme_idx], lh_ext)
        vfc_labels = vfc.predict(10)

        cluster_lh = {c: lh_ext[vfc_labels == c].mean() for c in np.unique(vfc_labels)}
        enriched_cl = max(cluster_lh, key=cluster_lh.get)
        depleted_cl = min(cluster_lh, key=cluster_lh.get)

        X_ext = X_raw[extreme_idx]
        enriched = vfc_labels == enriched_cl
        depleted = vfc_labels == depleted_cl

        activities, pvalues = [], []
        for i in range(adata.n_vars):
            _, pval = ranksums(X_ext[enriched, i], X_ext[depleted, i])
            activities.append(abs(X_ext[enriched, i].mean() - X_ext[depleted, i].mean()))
            pvalues.append(pval)

        # likelihood-weighted pseudotime for metric computation
        self.pseudotime = sum(c * likelihoods.iloc[:, i].values for i, c in enumerate(conditions))
        self.adata = adata
        return self._align_to_valid_genes(
            pd.DataFrame({"activity": activities, "pvalue": pvalues}, index=adata.var_names)
            .sort_values("activity", ascending=False)
        )


# ---------------------------------------------------------------------------
# Milo
# ---------------------------------------------------------------------------

class MiloMethod(TrajectoryMethod):
    """
    Milo differential abundance on a KNN graph.

    Uses ``ordinal_condition`` as a continuous predictor, then ranks genes by
    DE between DA-enriched and DA-depleted cells.
    """

    def __init__(self, data, input_type: str = "harmony", k: int = 50,
                 prop: float = 0.05, **kwargs):
        super().__init__(data, input_type=input_type)
        self.k = k
        self.prop = prop

    def get_gene_ranks(self) -> pd.DataFrame:
        try:
            import pertpy as pt
        except ImportError:
            raise ImportError("pertpy not installed: pip install pertpy")
        import scanpy as sc
        from scipy.stats import ranksums
        from sklearn.decomposition import PCA

        print("  Running Milo…")
        adata = self.adata.copy()
        X = self._get_input()
        X_raw = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)

        if self.input_type == "all_genes":
            X = PCA(n_components=30, random_state=0).fit_transform(X)

        adata.obsm["X_milo"] = X
        sc.pp.neighbors(adata, use_rep="X_milo", n_neighbors=self.k, random_state=0)

        milo = pt.tl.Milo()
        mdata = milo.load(adata)
        milo.make_nhoods(mdata["rna"], prop=self.prop)
        mdata = milo.count_nhoods(mdata, sample_col="sample")

        # da_nhoods requires the design variable to be sample-level
        # (one unique value per sample). Map each sample to its ordinal condition.
        sample_cond = (
            adata.obs.groupby("sample")["ordinal_condition"].first().astype(float)
        )
        mdata["rna"].obs["condition"] = mdata["rna"].obs["sample"].map(sample_cond)
        milo.da_nhoods(mdata, design="~condition", solver="edger")

        nhood_logfc = mdata["milo"].var["logFC"].values
        nhood_fdr = mdata["milo"].var.get(
            "SpatialFDR", mdata["milo"].var.get("FDR", mdata["milo"].var["PValue"])
        ).values
        nhood_matrix = mdata["rna"].obsm["nhoods"]

        sig = nhood_fdr < 0.1
        enriched_nhoods = sig & (nhood_logfc > 0)
        depleted_nhoods = sig & (nhood_logfc < 0)

        # Per-cell DA score
        sums = np.array(nhood_matrix.sum(axis=1)).ravel()
        cell_da = np.array((nhood_matrix @ nhood_logfc)).ravel()
        cell_da = np.where(sums > 0, cell_da / sums, 0.0)

        enriched_cells = np.array(nhood_matrix[:, enriched_nhoods].sum(axis=1)).ravel() > 0
        depleted_cells = np.array(nhood_matrix[:, depleted_nhoods].sum(axis=1)).ravel() > 0

        if enriched_cells.sum() == 0 or depleted_cells.sum() == 0:
            print("    No clear DA groups — falling back to DA-score correlation")
            activities, pvalues = [], []
            for i in range(adata.n_vars):
                corr, pval = spearmanr(X_raw[:, i], cell_da)
                activities.append(abs(corr))
                pvalues.append(pval)
        else:
            print(f"    {enriched_cells.sum()} enriched vs {depleted_cells.sum()} depleted cells")
            activities, pvalues = [], []
            for i in range(adata.n_vars):
                _, pval = ranksums(X_raw[enriched_cells, i], X_raw[depleted_cells, i])
                activities.append(abs(X_raw[enriched_cells, i].mean() - X_raw[depleted_cells, i].mean()))
                pvalues.append(pval)

        self.pseudotime = cell_da
        self.adata = adata
        return self._align_to_valid_genes(
            pd.DataFrame({"activity": activities, "pvalue": pvalues}, index=adata.var_names)
            .sort_values("activity", ascending=False)
        )


# ---------------------------------------------------------------------------
# PseudobulkDESeq2
# ---------------------------------------------------------------------------

class PseudobulkDESeq2Method(TrajectoryMethod):
    """
    Pseudobulk DESeq2 Wald tests.

    Aggregates raw integer counts per (sample × ordinal_condition) pseudobulk,
    then runs Wald tests for each stage vs stage 0.
    Gene activity = max |Wald stat| across contrasts.

    Raw counts are read from the ``counts`` layer of the h5ad.  If no such
    layer exists, ``X`` is used with a warning (results may be unreliable).
    """

    def _build_pseudobulk(self):
        import scipy.sparse as sp

        adata = self.adata

        if "counts" in adata.layers:
            X = adata.layers["counts"]
            genes = adata.var_names
        else:
            print("    Warning: no 'counts' layer — using X (may not be integer counts)")
            X = adata.X
            genes = adata.var_names

        # Restrict to genes with a valid ranking to keep the matrix manageable
        valid = self.data.gene_ranking.index[self.data.gene_ranking["SNR"].notna()]
        mask = genes.isin(valid)
        if mask.sum() < len(genes):
            print(f"    Subsetting from {len(genes)} to {mask.sum()} ranked genes")
            X = X[:, mask]
            genes = genes[mask]

        X = sp.csr_matrix(
            np.asarray(X.todense() if hasattr(X, "todense") else X, dtype=np.float32)
        )

        obs = adata.obs[["sample", "ordinal_condition"]].copy()
        obs = obs[obs["ordinal_condition"] >= 0]
        pos = np.array([adata.obs_names.get_loc(i) for i in obs.index])
        X = X[pos]

        rows, meta = [], []
        for (donor, stage), grp in obs.groupby(["sample", "ordinal_condition"]):
            idx = obs.index.get_indexer(grp.index)
            rows.append(np.asarray(X[idx].sum(axis=0)).ravel())
            meta.append({"sample": donor, "stage": int(stage),
                         "stage_str": f"s{int(stage)}"})

        pb = pd.DataFrame(rows, columns=genes).astype(int)
        pb_meta = pd.DataFrame(meta)
        ids = [f"{r['sample']}_s{r['stage']}" for _, r in pb_meta.iterrows()]
        pb.index = pb_meta.index = ids
        pb = pb.loc[:, (pb > 0).sum() >= 2]
        print(f"    {pb.shape[0]} pseudobulk profiles × {pb.shape[1]} genes")
        return pb, pb_meta

    def get_gene_ranks(self) -> pd.DataFrame:
        try:
            from pydeseq2.dds import DeseqDataSet
            from pydeseq2.ds import DeseqStats
        except ImportError:
            raise ImportError("pydeseq2 not installed: pip install pydeseq2")

        print("  Building pseudobulk profiles…")
        pb, pb_meta = self._build_pseudobulk()
        stages = sorted(pb_meta["stage"].unique())
        if 0 not in stages or len(stages) < 2:
            raise ValueError("Need ordinal_condition levels including 0 and at least one other")

        non_ref = [s for s in stages if s != 0]
        print(f"  Running DESeq2 Wald tests ({len(non_ref)} contrasts vs s0)…")

        dds = DeseqDataSet(
            counts=pb, metadata=pb_meta[["stage_str"]],
            design_factors="stage_str", ref_level=["stage_str", "s0"],
            quiet=True, n_cpus=4,
        )
        dds.deseq2()

        stat_cols, padj_cols = {}, {}
        for s in non_ref:
            st = DeseqStats(dds, contrast=["stage_str", f"s{s}", "s0"], quiet=True)
            st.summary()
            stat_cols[s] = st.results_df["stat"].abs().fillna(0.0)
            padj_cols[s] = st.results_df["padj"].fillna(1.0)

        activity = pd.DataFrame(stat_cols).max(axis=1)
        padj = (pd.DataFrame(padj_cols).min(axis=1) * len(non_ref)).clip(upper=1.0)

        return self._align_to_valid_genes(
            pd.DataFrame({"activity": activity, "pvalue": padj})
            .sort_values("activity", ascending=False)
        )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

METHODS: dict[str, type[TrajectoryMethod]] = {
    "TBR":              TBRMethod,
    "PAGA":             PAGAMethod,
    "DPT":              DPTMethod,
    "Palantir":         PalantirMethod,
    "ANOVA":            ANOVAMethod,
    "Wilcoxon":         WilcoxonMethod,
    "HiDDEN":           HiDDENMethod,
    "MELD":             MELDMethod,
    "Milo":             MiloMethod,
    "PseudobulkDESeq2": PseudobulkDESeq2Method,
}


def get_method(name: str, data, input_type: str = "all_genes",
               extra_kwargs: dict | None = None) -> TrajectoryMethod:
    if name not in METHODS:
        raise ValueError(f"Unknown method {name!r}. Available: {list(METHODS)}")
    return METHODS[name](data, input_type=input_type, **(extra_kwargs or {}))


# ---------------------------------------------------------------------------
# Per-method output writers
# ---------------------------------------------------------------------------

def _write_status(out_dir: Path, method_name: str, implemented: bool,
                   error: str | None) -> None:
    with open(out_dir / f"{method_name}_status.json", "w") as f:
        json.dump({"implemented": implemented, "error": error}, f)


def _write_gene_ranking(out_dir: Path, method_name: str, gene_ranks: pd.DataFrame) -> None:
    out = gene_ranks[["activity", "pvalue"]].copy()
    out.index.name = "gene"
    out.reset_index().to_csv(out_dir / f"{method_name}_gene_ranking.csv", index=False)


def _write_method_adata(out_dir: Path, method_name: str, method: TrajectoryMethod) -> None:
    """Write obs (incl. pseudotime) + obsm, with an empty var so X stays tiny."""
    import anndata as ad

    adata = getattr(method, "adata", method.data.adata)
    obs = pd.DataFrame(index=adata.obs_names)
    pseudotime = getattr(method, "pseudotime", None)
    if pseudotime is not None:
        obs["pseudotime"] = np.asarray(pseudotime)

    out = ad.AnnData(X=np.empty((adata.n_obs, 0), dtype=np.float32), obs=obs)
    for key, value in adata.obsm.items():
        out.obsm[key] = value
    ad.settings.allow_write_nullable_strings = True
    out.write_h5ad(out_dir / f"{method_name}_adata.h5ad")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="Run one trajectory method as a standalone process")
    p.add_argument("--config", required=True, help="Path to YAML config file")
    p.add_argument("--method", required=True, choices=list(METHODS))
    p.add_argument("--input_type", default="all_genes", choices=["all_genes", "harmony"])
    p.add_argument("--output_dir", default="outputs")
    return p.parse_args()


def main():
    import yaml

    from data_loader import load_data

    args = _parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    config_name = Path(args.config).stem

    out_dir = Path(args.output_dir) / config_name / args.input_type
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_data(
        cfg["data"]["h5ad_path"],
        cfg["data"]["pkl_path"],
    )

    method = get_method(args.method, data, input_type=args.input_type)

    print(f"\n{'='*60}\n  {args.method}  ({args.input_type})\n{'='*60}")
    try:
        gene_ranks = method.get_gene_ranks()
    except (ImportError, NotImplementedError) as exc:
        print(f"  Skipped: {exc}")
        _write_status(out_dir, args.method, implemented=False, error=str(exc))
        return

    _write_gene_ranking(out_dir, args.method, gene_ranks)
    _write_method_adata(out_dir, args.method, method)
    _write_status(out_dir, args.method, implemented=True, error=None)
    print(f"  Wrote outputs to {out_dir}")


if __name__ == "__main__":
    main()
