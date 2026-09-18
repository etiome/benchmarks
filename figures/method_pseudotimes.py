# For each method that produces a pseudotime, shows two panels:
# - **Top**: scatter of pseudotime vs iEP score with linear fit
# - **Bottom**: KNN-smoothed stacked bar of cell condition proportions along pseudotime
#
# Reads `method_production` from the latest `*_manuscript_data.pkl` in `results/`.


import glob
import os
import pickle
import math
import numpy as np
import matplotlib.pyplot as plt

RESULTS_DIR = '../outputs/results'
PLOTS_DIR = '../outputs/plots'
os.makedirs(PLOTS_DIR, exist_ok=True)

_tab10 = plt.cm.tab10.colors
METHOD_COLORS = {
    'TBR':      '#E41A1C', 'PAGA':     _tab10[0], 'DPT':      _tab10[1],
    'Palantir': _tab10[2], 'HiDDEN':   _tab10[5], 'MELD':     _tab10[6],
    'Milo':     _tab10[7],
}
METHOD_ORDER = ['TBR', 'PAGA', 'DPT', 'Palantir', 'HiDDEN', 'MELD', 'Milo']

# Etiome 6-stage palette
ETIOME_6 = [
    (0.416, 0.631, 0.643), (0.360, 0.793, 0.437), (0.737, 0.918, 0.329),
    (0.919, 0.922, 0.283), (0.927, 0.721, 0.236), (0.933, 0.486, 0.188),
]


def load_latest_pkl(input_type='harmony'):
    hits = glob.glob(os.path.join(RESULTS_DIR, f'*_{input_type}_manuscript_data.pkl'))
    if not hits:
        raise FileNotFoundError(f'No pkl for input_type={input_type} in {RESULTS_DIR}')
    path = max(hits, key=os.path.getmtime)
    print(f'Loading: {os.path.basename(path)}')
    with open(path, 'rb') as f:
        return pickle.load(f)


def knn_proportions(pseudotime, labels, n_pts=80, k=None):
    """KNN-smoothed proportion of each label category along pseudotime."""
    valid = np.isfinite(pseudotime)
    pt    = pseudotime[valid]
    lb    = np.array(labels)[valid]
    cats  = sorted(set(lb))
    if k is None:
        k = max(int(len(pt) / len(cats) * 0.1), 50)
    x_pts = np.linspace(pt.min(), pt.max(), n_pts)
    props = np.zeros((len(cats), n_pts))
    for j, x in enumerate(x_pts):
        dists = np.abs(pt - x)
        nn = np.argpartition(dists, min(k, len(pt)-1))[:k]
        for ci, cat in enumerate(cats):
            props[ci, j] = (lb[nn] == cat).mean()
    return x_pts, cats, props


INPUT_TYPE = 'harmony'  # change to 'all_genes' if preferred

data = load_latest_pkl(INPUT_TYPE)
mp = data.get('method_production', {})

if not mp:
    print('No method_production in this pkl — rerun the benchmark with the updated evaluate.py')
else:
    methods = [m for m in METHOD_ORDER if m in mp]
    methods += [m for m in mp if m not in METHOD_ORDER]
    print(f'Methods with pseudotime: {methods}')


n = len(methods)
ncols = 3
nrows = math.ceil(n / ncols)

fig = plt.figure(figsize=(ncols * 4, nrows * 3.5))
outer = fig.add_gridspec(nrows, ncols, hspace=0.6, wspace=0.4)

has_iep = any('iep_score' in mp[m]['obs'] for m in methods)

leg_h, leg_l = [], []

for idx, method in enumerate(methods):
    r, c = divmod(idx, ncols)
    inner = outer[r, c].subgridspec(2, 1, hspace=0.05, height_ratios=[1.5, 1])
    ax_s = fig.add_subplot(inner[0])
    ax_b = fig.add_subplot(inner[1])

    prod = mp[method]
    pt   = np.array(prod['pseudotime'], dtype=float)
    obs  = prod['obs']
    iep_corr = prod.get('iep_correlation')
    color = METHOD_COLORS.get(method, '#888888')

    # --- scatter: pseudotime vs iEP score ---
    if has_iep and 'iep_score' in obs:
        iep = np.array(obs['iep_score'], dtype=float)
        fin = np.isfinite(pt) & np.isfinite(iep)
        ax_s.scatter(pt[fin], iep[fin], s=1, alpha=0.1, color=color, rasterized=True)
        if iep_corr is None and fin.sum() > 1:
            from scipy.stats import pearsonr
            iep_corr, _ = pearsonr(pt[fin], iep[fin])
        if iep_corr is not None and np.isfinite(iep_corr):
            xs = np.array([pt[fin].min(), pt[fin].max()])
            z  = np.polyfit(pt[fin], iep[fin], 1)
            ax_s.plot(xs, np.poly1d(z)(xs), 'k--', linewidth=0.8)
            ax_s.text(0.05, 0.90, f'r={iep_corr:.2f}',
                      transform=ax_s.transAxes, fontsize=6)
        ax_s.set_ylabel('iEP score', fontsize=6)
    else:
        ax_s.set_visible(False)

    ax_s.set_title(method, fontsize=8)
    ax_s.tick_params(labelbottom=False, labelsize=5)
    ax_s.spines['top'].set_visible(False)
    ax_s.spines['right'].set_visible(False)

    # --- stackplot: cell condition proportions ---
    cond_key = 'cell_condition' if 'cell_condition' in obs else 'ordinal_condition'
    if cond_key in obs:
        labels_arr = np.array(obs[cond_key])
        fin_pt = np.isfinite(pt)
        x_pts, cats, props = knn_proportions(pt[fin_pt], labels_arr[fin_pt])
        n_cats = len(cats)
        colors = ETIOME_6[:n_cats] if n_cats <= 6 else plt.cm.tab10.colors[:n_cats]
        h = ax_b.stackplot(x_pts, props, colors=colors,
                           labels=[str(c) for c in cats],
                           edgecolor='white', linewidth=0.3)
        if not leg_h:
            leg_h, leg_l = ax_b.get_legend_handles_labels()
    ax_b.set_yticks([])
    ax_b.margins(x=0, y=0)
    ax_b.set_xlabel('pseudotime', fontsize=6)
    ax_b.set_ylabel('proportion', fontsize=6)
    ax_b.tick_params(labelsize=5)
    ax_b.spines['top'].set_visible(False)
    ax_b.spines['right'].set_visible(False)

# Fill empty slots with legend
for idx in range(len(methods), nrows * ncols):
    r, c = divmod(idx, ncols)
    inner = outer[r, c].subgridspec(2, 1)
    ax_top = fig.add_subplot(inner[0])
    ax_bot = fig.add_subplot(inner[1])
    ax_bot.set_visible(False)
    ax_top.axis('off')
    if idx == len(methods) and leg_h:
        ax_top.legend(leg_h, leg_l, loc='center', frameon=False,
                      ncol=2, title='condition', fontsize=7)

fig.suptitle(f'Per-method pseudotime ({INPUT_TYPE})', fontsize=10)
fig.savefig(os.path.join(PLOTS_DIR, f'method_pseudotimes_{INPUT_TYPE}.png'), dpi=300)
plt.close(fig)
