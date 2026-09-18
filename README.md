# TBR Trajectory Benchmarks

Standalone replication package for the trajectory-method benchmarks reported in
[ADD LINK HERE WHEN AVAILABLE].

Given a single-cell dataset characterizing a biological process ([Biddy, et al. 2018](https://www.nature.com/articles/s41586-018-0744-4)), we compare ten
methods on their ability to rank genes that have known association with that process.

The quality of each method's ranking is assessed by **GSEA-style normalized enrichment scores** (NES) against a curated
set of 68 markers known to change during induced endoderm progenitors -> mouse embryonic fibroblasts reprogramming.

## Requirements

* Docker
* 32GB+ of RAM
* 5GB+ of disk space

## Quick start (Docker)

Pull the latest published image from GHCR:

```bash
docker pull ghcr.io/etiome/benchmarks:latest
```

Run the full benchmark, mapping a local `outputs/` folder to the
container's output directory so every result lands on your host:

```bash
docker run --rm -v "$(pwd)/outputs:/benchmarks/outputs" ghcr.io/etiome/benchmarks:latest
```

Once finished, the `outputs/` directory will contain the following:

* Gene ranking for every method
* Assignment of cells to disease progression coordinates (trajectory reconstruction methods only)
* Summary tables showing performance metrics
* Plots that reproduce manuscript figures

---

### Methods compared

| Method | Type |
|---|---|
| **TBR** | Temporal Biodynamic Representation |
| PAGA | Partition-based graph abstraction + diffusion pseudotime |
| DPT | Diffusion pseudotime |
| Palantir | Palantir pseudotime |
| HiDDEN | Logistic-regression label refinement |
| MELD | Manifold enhancement of latent dimensions |
| Milo | KNN-graph differential abundance |
| ANOVA | One-way ANOVA F-statistic (non-trajectory) |
| Wilcoxon | Wilcoxon rank-sum test (non-trajectory) |
| PseudobulkDESeq2 | Pseudobulk DESeq2 Wald test (non-trajectory) |

Each method is run with two input representations:
- **`all_genes`** — operates on the full normalised gene matrix
- **`harmony`** — operates on Harmony-corrected PCA embeddings

### Metrics reported

| Metric | Description |
|---|---|
| NES | GSEA normalised enrichment score against the ground-truth gene set |
| IEP Correlation | Pearson r between pseudotime and per-cell iEP score |
| Concordance Index | Fraction of sample pairs where pseudotime order matches disease stage |
| Transcriptional Continuity | Weighted autocorrelation of cell distances along pseudotime |

---

## Repository structure

```
tbr_benchmarks/
├── data/             # Input data wrangled from the Biddy, et al. 2018 study
├── configs/          # Orchestration parameters and ground truth genes
├── generate.py       # Generates simulated patients from the original single cell data
├── data_loader.py    # Loads input data and makes it accessible as conventient Python objects
├── methods.py        # Method implementations
├── metrics.py        # IEP correlation, concordance index, transcriptional continuity
├── evaluate.py       # Scoring of individual method outputs
├── plotting.py       # NES curve comparison plot
├── run.sh            # Single entry-point: runs both input types
├── requirements.txt  # Python dependencies
└── Dockerfile        # Runtime environment definition
```
