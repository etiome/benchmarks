# Trajectory Method Benchmarks — NES Enrichment Curves
#
# GSEA-style enrichment curves comparing all trajectory methods against the
# hepatocyte reprogramming ground-truth gene set.
#
# Reads the latest `*_manuscript_data.pkl` from `outputs/` for each input type.

import glob
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

RESULTS_DIR = '../outputs/results'
PLOTS_DIR = '../outputs/plots'
os.makedirs(PLOTS_DIR, exist_ok=True)

# Fixed per-method colours (matching the manuscript)
_tab10 = plt.cm.tab10.colors
METHOD_COLORS = {
    'TBR':              '#E41A1C',
    'PAGA':             _tab10[0],
    'DPT':              _tab10[1],
    'Palantir':         _tab10[2],
    'ANOVA':            _tab10[4],
    'Wilcoxon':         _tab10[5],
    'HiDDEN':           _tab10[6],
    'MELD':             _tab10[7],
    'Milo':             _tab10[8],
    'PseudobulkDESeq2': _tab10[9],
}


def load_latest_pkl(input_type):
    """Return (method_names, enrichment_data) from the most recent pkl for input_type."""
    hits = glob.glob(os.path.join(RESULTS_DIR, f'*_{input_type}_manuscript_data.pkl'))
    if not hits:
        raise FileNotFoundError(f'No pkl found for input_type={input_type} in {RESULTS_DIR}')
    path = max(hits, key=os.path.getmtime)
    print(f'  {input_type}: {os.path.basename(path)}')
    with open(path, 'rb') as f:
        d = pickle.load(f)
    ec = d['enrichment_comparison']
    return ec['method_names'], ec['enrichment_data']


def plot_enrichment_panel(ax, method_names, enrichment_data, title):
    """Plot GSEA NES curves, TBR on top in red."""
    # Sort ascending by NES so TBR plots last (on top)
    order = sorted(range(len(method_names)), key=lambda i: enrichment_data[i][1])
    for i in order:
        name = method_names[i]
        curve, nes = enrichment_data[i]
        is_tbr = name == 'TBR'
        ax.plot(
            np.arange(len(curve)), curve,
            color=METHOD_COLORS.get(name, '#888888'),
            linewidth=1.5 if is_tbr else 0.8,
            zorder=3 if is_tbr else 2,
            label=f'{name} (NES={nes:.2f})',
        )
    ax.axhline(0, color='black', linewidth=0.5, linestyle='--', alpha=0.4)
    ax.set_title(title)
    ax.set_xlabel('Gene rank')
    ax.set_ylabel('NES')
    ax.legend(loc='upper right', fontsize=5, framealpha=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


fig, axes = plt.subplots(1, 2, figsize=(12, 4))

for ax, input_type in zip(axes, ['all_genes', 'harmony']):
    try:
        names, data = load_latest_pkl(input_type)
        plot_enrichment_panel(ax, names, data,
                              f'Mouse reprogramming target enrichment ({input_type})')
    except FileNotFoundError as e:
        ax.text(0.5, 0.5, str(e), transform=ax.transAxes, ha='center', va='center')
        ax.set_title(input_type)

fig.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, 'enrichment_curves_by_input_type.png'), dpi=300)
plt.close(fig)


# Allow each competitor to select PCA or Harmony, whichever produces the best results to
#  demonstrate that TBR still outperforms all of them
try:
    names_ag, data_ag = load_latest_pkl('all_genes')
    names_ha, data_ha = load_latest_pkl('harmony')

    ag = dict(zip(names_ag, data_ag))
    ha = dict(zip(names_ha, data_ha))
    all_methods = sorted(set(names_ag) | set(names_ha))

    best_names, best_data = [], []
    for m in all_methods:
        a, h = ag.get(m), ha.get(m)
        if a is None and h is None:
            continue
        elif a is None:
            best_names.append(m); best_data.append(h)
        elif h is None:
            best_names.append(m); best_data.append(a)
        else:
            best_names.append(m); best_data.append(a if a[1] >= h[1] else h)

    fig, ax = plt.subplots(figsize=(6, 4))
    plot_enrichment_panel(ax, best_names, best_data,
                          'Mouse reprogramming target enrichment (best of all_genes / harmony)')
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, 'enrichment_curves_best_of.png'), dpi=300)
    plt.close(fig)

except FileNotFoundError as e:
    print(f'Skipping best-of plot: {e}')
