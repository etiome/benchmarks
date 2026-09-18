"""
data_loader.py — load the benchmark inputs from plain files.
"""

from __future__ import annotations

import lzma
import pickle
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

# Genes below these thresholds carry no usable trajectory signal and are
# dropped on load to avoid OOM in downstream methods.
MIN_CELLS_NONZERO = 100


class BenchmarkData:
    """Container for all benchmark inputs."""

    def __init__(
        self,
        h5ad_path: str,
        pkl_path: str,
    ):
        # --- cell × gene expression ---
        print(f"Loading h5ad: {h5ad_path}")
        self.adata = ad.read_h5ad(h5ad_path)
        _require_obs_cols(self.adata, ["sample", "ordinal_condition"])

        # --- drop genes with no usable signal (avoids OOM downstream) ---
        n_genes_before = self.adata.n_vars
        self.adata = _filter_low_expression_genes(self.adata, min_cells_nonzero=MIN_CELLS_NONZERO)
        print(
            f"  Filtered genes: {n_genes_before} -> {self.adata.n_vars} "
            f"(dropped {n_genes_before - self.adata.n_vars} with max expression == 0 "
            f"or non-zero in < {MIN_CELLS_NONZERO} cells)"
        )

        with lzma.open(pkl_path, "rb") as f:
            data = pickle.load(f)

        emb_df = data["embedding"].set_index("cell")
        if "embedding" not in emb_df.columns:
            raise ValueError(f"pkl_path['embedding'] must have an 'embedding' column; got {emb_df.columns.tolist()}")
        shared = self.adata.obs_names.intersection(emb_df.index)
        if len(shared) < self.adata.n_obs:
            print(f"  Warning: {self.adata.n_obs - len(shared)} cells in h5ad not found in embedding data — subsetting.")
            self.adata = self.adata[shared].copy()
        self.embedding: pd.Series = emb_df.loc[self.adata.obs_names, "embedding"]

        ranks = data["gene_ranking"].set_index("gene")
        if "SNR" not in ranks.columns:
            raise ValueError(
                f"pkl_path['gene_ranking'] must have an 'SNR' column; got {ranks.columns.tolist()}"
            )
        self.gene_ranking: pd.DataFrame = ranks

        print(
            f"Loaded: {self.adata.n_obs} cells × {self.adata.n_vars} genes | "
            f"{self.gene_ranking['SNR'].notna().sum()} ranked genes"
        )


def _require_obs_cols(adata: ad.AnnData, cols: list[str]) -> None:
    missing = [c for c in cols if c not in adata.obs.columns]
    if missing:
        raise ValueError(f"h5ad obs is missing required columns: {missing}")


def _filter_low_expression_genes(adata: ad.AnnData, min_cells_nonzero: int) -> ad.AnnData:
    """Drop genes with zero max expression or non-zero expression in too few cells.
    """
    X = adata.X
    if sp.issparse(X):
        max_per_gene = np.asarray(X.max(axis=0).toarray()).ravel()
        n_cells_nonzero = X.getnnz(axis=0)
    else:
        max_per_gene = np.asarray(X).max(axis=0).ravel()
        n_cells_nonzero = np.asarray((X != 0).sum(axis=0)).ravel()

    keep = (max_per_gene > 0) & (n_cells_nonzero >= min_cells_nonzero)
    filtered = adata[:, keep].copy()
    return filtered

def load_data(
    h5ad_path: str,
    pkl_path: str,
) -> BenchmarkData:
    """Load benchmark inputs and return a BenchmarkData object."""
    return BenchmarkData(h5ad_path, pkl_path)
