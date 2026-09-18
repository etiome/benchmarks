"""
Benchmark evaluation — loads per-method outputs and scores them.

Usage (CLI)::

    python evaluate.py --config configs/Mouse_Reprogramming.yaml \\
                       --input_type all_genes

For each method, reads the outputs written by ``methods.py`` and
scores each ranking with gseapy ``prerank`` against the ground-truth gene set.
Saves a CSV summary and a ``*_manuscript_data.pkl`` containing the NES curves
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path

import anndata as ad
import gseapy as gp
import numpy as np
import pandas as pd
import yaml

from data_loader import load_data
from methods import METHODS
from metrics import iep_correlation, concordance_index, transcriptional_continuity


# ---------------------------------------------------------------------------
# Per-method evaluation
# ---------------------------------------------------------------------------

def evaluate_method(
    name: str,
    data,
    ground_truth_genes: list[str],
    input_type: str = "all_genes",
    method_output_dir: str = "outputs",
    config_name: str = "benchmark",
) -> dict:
    """
    Load one method's outputs (written by ``methods.py``) and score it.

    Returns
    -------
    dict with keys: method, NES, gene_ranks, enrichment_data, implemented
    """
    print(f"\n{'='*60}")
    print(f"  {name}  ({input_type})")
    print(f"{'='*60}")

    out_dir = Path(method_output_dir) / config_name / input_type

    with open(out_dir / f"{name}_status.json") as f:
        status = json.load(f)

    if not status["implemented"]:
        print(f"  Skipped: {status['error']}")
        return {"method": name, "NES": float("nan"),
                "iep_correlation": float("nan"),
                "concordance_index": float("nan"),
                "transcriptional_continuity": float("nan"),
                "gene_ranks": None, "enrichment_data": None,
                "pseudotime": None, "obs": None, "implemented": False}

    gene_ranks = pd.read_csv(out_dir / f"{name}_gene_ranking.csv", index_col="gene")[
        ["activity", "pvalue"]
    ]

    # Recover pseudotime + embedding saved by methods.py, reindexed against
    # the shared BenchmarkData loaded once by this script.
    pt = None
    obsm: dict = {}
    h5ad_path = out_dir / f"{name}_adata.h5ad"
    if h5ad_path.exists():
        method_adata = ad.read_h5ad(h5ad_path)[data.adata.obs_names]
        if "pseudotime" in method_adata.obs.columns:
            pt = method_adata.obs["pseudotime"].values
        obsm = dict(method_adata.obsm)

    obs = data.adata.obs

    iep_corr = iep_correlation(pt, obs) if pt is not None else float("nan")
    ci       = concordance_index(pt, obs) if pt is not None else float("nan")

    print(f"  IEP correlation       : {iep_corr:.3f}" if not np.isnan(iep_corr) else "  IEP correlation       : n/a")
    print(f"  Concordance index     : {ci:.3f}"       if not np.isnan(ci)       else "  Concordance index     : n/a")

    tc = float("nan")
    if pt is not None:
        print("  Computing transcriptional continuity…")
        adata_for_tc = data.adata
        if obsm:
            adata_for_tc = data.adata.copy()
            for key, value in obsm.items():
                adata_for_tc.obsm[key] = value
        tc = transcriptional_continuity(pt, adata_for_tc)
        print(f"  Transcriptional cont. : {tc:.3f}" if not np.isnan(tc) else "  Transcriptional cont. : n/a")

    enrichment_data = _run_gsea(gene_ranks["activity"], ground_truth_genes)
    nes = enrichment_data[1] if enrichment_data else float("nan")
    print(f"  NES = {nes:.3f}" if not np.isnan(nes) else "  NES = n/a")

    return {
        "method": name,
        "NES": nes,
        "iep_correlation": iep_corr,
        "concordance_index": ci,
        "transcriptional_continuity": tc,
        "gene_ranks": gene_ranks,
        "enrichment_data": enrichment_data,
        "pseudotime": pt,
        "obs": data.adata.obs[
            [c for c in ["iep_score", "cell_condition", "ordinal_condition", "sample"]
             if c in data.adata.obs.columns]
        ].to_dict(orient="list") if pt is not None else None,
        "implemented": True,
    }


def _run_gsea(ranks: pd.Series, ground_truth_genes: list[str]):
    """Run gseapy prerank and return (NES_curve, nes) or None on failure."""
    if ranks.nunique() / max(len(ranks), 1) < 0.001:
        print("  Skipping GSEA: gene ranks are all identical")
        return None
    try:
        pre_res = gp.prerank(
            rnk=ranks,
            gene_sets={"genes": ground_truth_genes},
            threads=4,
            min_size=1,
            max_size=10_000,
            permutation_num=1000,
            outdir=None,
            seed=6,
            verbose=False,
        )
        result = pre_res.results["genes"]
        es, nes = result["es"], result["nes"]
        RES = np.array(result["RES"])
        NES_curve = RES / (es / nes) if (nes != 0 and es != 0) else RES
        return (NES_curve, nes)
    except Exception as exc:
        print(f"  gseapy failed: {exc}")
        return None


# ---------------------------------------------------------------------------
# Full benchmark run
# ---------------------------------------------------------------------------

def run_benchmark(
    h5ad_path: str,
    pkl_path: str,
    ground_truth_genes: list[str],
    methods: list[str],
    input_type: str = "all_genes",
    output_dir: str = "outputs/results",
    config_name: str = "benchmark",
    method_output_dir: str = "outputs",
) -> pd.DataFrame:
    """
    Evaluate all methods and save results.
    """
    print(f"\n{'='*60}")
    print(f"Loading data…")
    data = load_data(h5ad_path, pkl_path)
    print(f"Methods      : {', '.join(methods)}")
    print(f"Input type   : {input_type}")
    print(f"Ground truth : {len(ground_truth_genes)} genes")
    print(f"{'='*60}\n")

    results = [
        evaluate_method(name, data, ground_truth_genes, input_type,
                        method_output_dir=method_output_dir, config_name=config_name)
        for name in methods
    ]

    # Summary DataFrame
    df = (
        pd.DataFrame([{
            "Method": r["method"],
            "NES": r["NES"],
            "IEP Correlation": r["iep_correlation"],
            "Concordance Index": r["concordance_index"],
            "Transcriptional Continuity": r["transcriptional_continuity"],
            "Implemented": r["implemented"],
        } for r in results])
        .sort_values("NES", ascending=False)
    )
    print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
    print(df.to_string(index=False))

    # Save
    os.makedirs(output_dir, exist_ok=True)
    stem = f"trajectory_benchmark_{config_name}_{input_type}"

    csv_path = Path(output_dir) / f"{stem}.csv"
    pkl_path = Path(output_dir) / f"{stem}_manuscript_data.pkl"

    df.to_csv(csv_path, index=False)
    _save_pkl(pkl_path, results)

    print(f"\nSaved CSV → {csv_path}")
    print(f"Saved pkl  → {pkl_path}")

    return df


def _save_pkl(pkl_path: Path, results: list[dict]) -> None:
    """Write the manuscript_data.pkl consumed by plotting scripts."""
    method_names, enrichment_data = [], []
    method_production = {}

    for r in results:
        if not r["implemented"]:
            continue
        if r["enrichment_data"] is not None:
            method_names.append(r["method"])
            enrichment_data.append(r["enrichment_data"])
        if r.get("pseudotime") is not None and r.get("obs") is not None:
            method_production[r["method"]] = {
                "pseudotime": np.array(r["pseudotime"]),
                "obs": r["obs"],
                "iep_correlation": r["iep_correlation"],
            }

    with open(pkl_path, "wb") as f:
        pickle.dump(
            {
                "enrichment_comparison": {
                    "method_names": method_names,
                    "enrichment_data": enrichment_data,
                },
                "method_production": method_production,
            },
            f,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="Trajectory benchmark evaluation")
    p.add_argument("--config", required=True, help="Path to YAML config file")
    p.add_argument("--input_type", default=None, choices=["all_genes", "harmony"],
                   help="Input type (overrides config)")
    p.add_argument("--methods", nargs="+", default=None,
                   help="Methods to run (overrides config)")
    p.add_argument("--output_dir", default=None,
                   help="Output directory (overrides config)")
    p.add_argument("--method_output_dir", default="outputs",
                   help="Root directory of per-method outputs written by methods.py")
    return p.parse_args()


def main():
    args = _parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    config_name = Path(args.config).stem
    params = cfg.get("parameters", {})

    h5ad_path    = cfg["data"]["h5ad_path"]
    pkl_path     = cfg["data"]["pkl_path"]
    ground_truth = cfg["ground_truth_genes"]

    methods    = args.methods    or params.get("methods_to_test", list(METHODS))
    input_type = args.input_type or params.get("input_type", "all_genes")
    output_dir = args.output_dir or params.get("output_dir", "outputs/results")

    run_benchmark(
        h5ad_path=h5ad_path,
        pkl_path=pkl_path,
        ground_truth_genes=ground_truth,
        methods=methods,
        input_type=input_type,
        output_dir=output_dir,
        config_name=config_name,
        method_output_dir=args.method_output_dir,
    )


if __name__ == "__main__":
    main()
