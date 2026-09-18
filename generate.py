"""
generate_mouse_reprogramming.py — generate the synthetic mouse reprogramming h5ad.

Creates artificial 'patient' samples from the Biddy 2018 mouse reprogramming
single-cell dataset by mixing cells drawn from a Gaussian distribution over
100 iEP-score bins, producing a smooth disease-progression spectrum.

Generation parameters:
  - seed=42
  - n_samples=30, cells_per_sample=1500
  - additive_noise_scale=0.2, multiplicative_noise_scale=0.4
  - n_genes_affected=0.7
  - sample_middle=True, sample_continuous=True, sigma_bins=12

Usage:
    python generate_mouse_reprogramming.py [--output_dir PATH]
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import scanpy as sc


# ---------------------------------------------------------------------------
# Synthetic generation
# ---------------------------------------------------------------------------

class SyntheticReprogramming:
    """
    Generate synthetic 'patient' samples from mouse reprogramming single-cell data.

    Samples are mixtures of cells drawn from a Gaussian distribution over
    100 iEP-score bins, producing a smooth disease-progression spectrum.
    """

    def __init__(self, adata, seed: int = 42):
        np.random.seed(seed)
        self.seed = seed
        self.adata = adata

        if "iep_score" not in adata.obs.columns:
            raise ValueError("adata must have 'iep_score' in .obs")

    def _assign_stages(self):
        scores = self.adata.obs["iep_score"].values
        stages = np.zeros(len(scores), dtype=int)
        stages[(scores >= 0.167) & (scores < 0.333)] = 1
        stages[(scores >= 0.333) & (scores < 0.5)]   = 2
        stages[(scores >= 0.5)   & (scores < 0.667)] = 3
        stages[(scores >= 0.667) & (scores < 0.833)] = 4
        stages[scores >= 0.833]                        = 5
        self.adata.obs["cell_condition"] = stages

    def _add_sample_noise(self, adata, additive_scale, multiplicative_scale, n_genes_affected):
        """Add sample-specific additive and multiplicative noise to HVGs.
        Exact replica of original implementation."""
        from scipy.sparse import csr_matrix

        X = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
        X = X.astype(np.float64)

        # Compute HVGs on the synthetic adata (matches original)
        print("Computing HVGs for noise addition...")
        sc.pp.highly_variable_genes(adata, n_top_genes=7500)
        hvg_indices = np.where(adata.var["highly_variable"].values)[0]
        n_hvg = len(hvg_indices)
        n_affected = int(n_genes_affected * n_hvg)

        samples = adata.obs["sample"].unique()
        print(f"Additive noise scale: {additive_scale}")
        print(f"Multiplicative noise scale: {multiplicative_scale}")
        print(f"HVGs: {n_hvg}")
        print(f"Genes affected: {n_affected}/{n_hvg} HVGs ({100*n_genes_affected:.1f}%)")

        for sample in samples:
            mask = adata.obs["sample"] == sample
            sample_data = X[mask]

            # Select random HVGs to affect
            affected_genes = np.random.choice(hvg_indices, size=n_affected, replace=False)

            # Multiplicative noise (lognormal)
            mult_factors = np.random.lognormal(mean=0, sigma=multiplicative_scale,
                                               size=n_affected)
            sample_data[:, affected_genes] *= mult_factors

            # Additive noise (Gaussian)
            add_noise = np.random.normal(
                loc=0,
                scale=additive_scale * sample_data[:, affected_genes].mean(),
                size=(sample_data.shape[0], n_affected)
            )
            sample_data[:, affected_genes] += add_noise

            # Ensure valid counts
            sample_data[sample_data < 0] = 0
            sample_data = np.floor(sample_data)
            X[mask] = sample_data

        adata.X = csr_matrix(X)

    def generate_samples(
        self,
        n_samples: int = 30,
        cells_per_sample: int = 1500,
        additive_noise_scale: float = 0.2,
        multiplicative_noise_scale: float = 0.4,
        n_genes_affected: float = 0.7,
        sigma_bins: float = 12.0,
    ):
        assert n_samples * cells_per_sample <= self.adata.n_obs, "Not enough cells"

        self._assign_stages()

        # Bin iEP scores into 100 bins
        iep_scores = self.adata.obs["iep_score"].values
        bins = np.linspace(0, 1, 101)
        bin_indices = np.clip(np.digitize(iep_scores, bins) - 1, 0, 99)

        # Pre-assign center bins — balanced across 6 conditions
        samples_per_cond = n_samples // 6
        remainder = n_samples % 6
        center_bins_list = []
        for cond in range(6):
            count = samples_per_cond + (1 if cond < remainder else 0)
            bin_range = range(cond * 17, (cond + 1) * 17) if cond < 5 else range(84, 100)
            for _ in range(count):
                center_bins_list.append(np.random.choice(bin_range))
        np.random.shuffle(center_bins_list)

        all_ids, all_cell_idx, all_cond = [], [], []
        available = np.arange(self.adata.n_obs)

        for i in range(n_samples):
            center_bin = center_bins_list[i]
            # Gaussian weights over bins
            bin_probs = np.exp(-0.5 * ((np.arange(100) - center_bin) / sigma_bins) ** 2)
            bin_probs /= bin_probs.sum()

            sample_idx = []
            for b in range(100):
                n_needed = int(np.round(bin_probs[b] * cells_per_sample))
                if n_needed == 0:
                    continue
                avail_b = available[bin_indices[available] == b]
                n_needed = min(n_needed, len(avail_b))
                if n_needed > 0:
                    sample_idx.extend(np.random.choice(avail_b, n_needed, replace=False))

            sample_idx = np.array(sample_idx)
            available = np.setdiff1d(available, sample_idx)

            cond = min(int(center_bin / 100.0 * 6), 5)
            all_ids.extend([f"Sample_{i:02d}"] * len(sample_idx))
            all_cell_idx.extend(sample_idx)
            all_cond.extend([cond] * len(sample_idx))

        adata_out = self.adata[all_cell_idx].copy()
        adata_out.obs["sample"] = all_ids
        adata_out.obs["condition"] = all_cond

        self._add_sample_noise(adata_out, additive_noise_scale, multiplicative_noise_scale, n_genes_affected)
        adata_out.obs["ordinal_condition"] = adata_out.obs["condition"]

        return adata_out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic mouse reprogramming h5ad")
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Output directory (default: data/)",
    )
    args = parser.parse_args()

    # --- Paths ---
    _here = Path(__file__).resolve().parent
    source_h5ad = _here / "data" / "Biddy_Preprocessed.h5ad"
    out_dir = Path(args.output_dir) if args.output_dir else _here / "data"
    out_path = out_dir / "Biddy_Patient_Snapshot.h5ad"

    # --- Load source (already has human gene names) ---
    print(f"Loading source h5ad: {source_h5ad}")
    adata = sc.read_h5ad(source_h5ad)
    print(f"  {adata.n_obs} cells × {adata.n_vars} genes")

    # Rename columns to match what the generation code expects
    if 'sample' in adata.obs.columns and 'batch' not in adata.obs.columns:
        adata.obs = adata.obs.rename(columns={"sample": "batch"})

    # --- Generate ---
    print("\nGenerating synthetic samples...")
    synth = SyntheticReprogramming(adata.copy(), seed=42)
    adata_synthetic = synth.generate_samples(
        n_samples=30,
        cells_per_sample=1500,
        additive_noise_scale=0.2,
        multiplicative_noise_scale=0.4,
        n_genes_affected=0.7,
        sigma_bins=12,
    )
    print(f"\nGenerated: {adata_synthetic.n_obs} cells × {adata_synthetic.n_vars} genes")

    # condition → string category
    adata_synthetic.obs["condition"] = adata_synthetic.obs["condition"].astype(str).astype("category")

    # --- Save ---
    os.makedirs(out_dir, exist_ok=True)
    print(f"\nSaving to: {out_path}")
    import anndata
    anndata.settings.allow_write_nullable_strings = True
    adata_synthetic.write_h5ad(out_path)
    print("Done.")

    return adata_synthetic, out_path


if __name__ == "__main__":
    main()
