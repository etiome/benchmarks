#!/usr/bin/env bash
# run.sh — replicate the Mouse Reprogramming benchmark
#
# Usage:
#   bash run.sh [CONFIG] [OUTPUT_DIR]
#
# Defaults:
#   CONFIG     = configs/Mouse_Reprogramming.yaml
#   OUTPUT_DIR = outputs/results
#
# Stage 0 — input generation:
#   Download data/Biddy_Preprocessed.h5ad from Zenodo if not already
#   present (verified against its md5 checksum), then generate.py builds
#   data/Biddy_Patient_Snapshot.h5ad from it.
#
# Stage 1 — per-method subprocess isolation:
#   For every (method x input_type) combination read from the config's
#   parameters.methods_to_test list, `methods.py` is launched as a
#   standalone subprocess. Each process's peak RAM is released when it
#   exits, and a crash in one method's process cannot affect the others.
#   Outputs land in outputs/{config_name}/{input_type}/.
#
# Stage 2 — evaluation:
#   Both input types used in the manuscript are then evaluated:
#     1. all_genes  — each method operates on the full gene matrix
#     2. harmony    — each method uses Harmony-corrected PCA embeddings
#
# For each evaluation run the script produces (consolidated under outputs/):
#   outputs/results/trajectory_benchmark_*_all_genes.csv
#   outputs/results/trajectory_benchmark_*_all_genes_manuscript_data.pkl
#   outputs/results/trajectory_benchmark_*_harmony.csv
#   outputs/results/trajectory_benchmark_*_harmony_manuscript_data.pkl
#
# Stage 3 — plot generation:
#   figures/enrichment_curves.py, figures/method_pseudotimes.py, and
#   figures/patient_snapshots.py each read from ../outputs/results (and,
#   for patient_snapshots.py, ../data) and save PNGs to outputs/plots/.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${1:-$SCRIPT_DIR/configs/Mouse_Reprogramming.yaml}"
METHOD_OUTPUT_DIR="$SCRIPT_DIR/outputs"
OUTPUT_DIR="${2:-$METHOD_OUTPUT_DIR/results}"

if [[ ! -f "$CONFIG" ]]; then
    echo "Error: config not found: $CONFIG"
    echo "Usage: bash run.sh [CONFIG] [OUTPUT_DIR]"
    exit 1
fi

echo "========================================"
echo "TBR Trajectory Benchmark"
echo "Config:     $CONFIG"
echo "Output dir: $OUTPUT_DIR"
echo "========================================"

cd "$SCRIPT_DIR"

echo ""
echo "--- Stage 0: fetch input data + generate snapshot ---"

BIDDY_H5AD="$SCRIPT_DIR/data/Biddy_Preprocessed.h5ad"
BIDDY_MD5="63cf28b672b681e95f6a03be170e7d7a"
if [[ -f "$BIDDY_H5AD" ]] && echo "$BIDDY_MD5  $BIDDY_H5AD" | md5sum -c - &>/dev/null; then
    echo "Using cached $BIDDY_H5AD"
else
    echo "Downloading Biddy_Preprocessed.h5ad from Zenodo..."
    curl -f -L --progress-bar -o "$BIDDY_H5AD" https://zenodo.org/records/22817389/files/Biddy_Preprocessed.h5ad
    echo "$BIDDY_MD5  $BIDDY_H5AD" | md5sum -c -
fi

python -u generate.py

# Read methods_to_test from the config so run.sh stays in sync with it.
mapfile -t METHODS < <(python -c "
import yaml, sys
with open(sys.argv[1]) as f:
    cfg = yaml.safe_load(f)
for m in cfg['parameters']['methods_to_test']:
    print(m)
" "$CONFIG")

INPUT_TYPES=("all_genes" "harmony")

echo ""
echo "--- Stage 1: per-method subprocess runs ---"
for input_type in "${INPUT_TYPES[@]}"; do
    for method in "${METHODS[@]}"; do
        echo ""
        echo ">>> method=$method input_type=$input_type"
        python -u methods.py --config "$CONFIG" --method "$method" \
            --input_type "$input_type" --output_dir "$METHOD_OUTPUT_DIR"
    done
done

echo ""
echo "--- Stage 2/2: Run 1/2: input_type=all_genes ---"
python -u evaluate.py --config "$CONFIG" --input_type all_genes \
    --output_dir "$OUTPUT_DIR" --method_output_dir "$METHOD_OUTPUT_DIR"

echo ""
echo "--- Stage 2/2: Run 2/2: input_type=harmony ---"
python -u evaluate.py --config "$CONFIG" --input_type harmony \
    --output_dir "$OUTPUT_DIR" --method_output_dir "$METHOD_OUTPUT_DIR"

echo ""
echo "--- Stage 3/3: plot generation ---"
FIGURES_DIR="$SCRIPT_DIR/figures"
for fig_script in enrichment_curves.py method_pseudotimes.py patient_snapshots.py; do
    echo ""
    echo ">>> figures/$fig_script"
    (cd "$FIGURES_DIR" && MPLBACKEND=Agg python -u "$fig_script")
done

echo ""
echo "========================================"
echo "Benchmark complete.  Results in: $OUTPUT_DIR"
echo "Plots in:            $METHOD_OUTPUT_DIR/plots"
echo "========================================"
