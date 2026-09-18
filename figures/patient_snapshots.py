# Mouse Reprogramming — Patient Snapshot Visualisation
#
# Reproduces the plots from the manuscript using the synthetic
# patient snapshot h5ad.
#
# **Plots produced:**
# 1. Sample composition — stacked horizontal bar showing iEP-stage proportions per sample
# 2. UMAP of synthetic data coloured by sample condition and cell condition
# 3. UMAP of the source (original Biddy 2018) data coloured by batch and iEP score
# 4. Cell condition proportions by sample condition (stacked bar)


import os
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

PLOTS_DIR = '../outputs/plots'
os.makedirs(PLOTS_DIR, exist_ok=True)

# Etiome palette (6 stages: MEF → iEP)
ETIOME_6 = [
    (0.416, 0.631, 0.643),  # stage 0 — MEF-like
    (0.360, 0.793, 0.437),  # stage 1
    (0.737, 0.918, 0.329),  # stage 2
    (0.919, 0.922, 0.283),  # stage 3
    (0.927, 0.721, 0.236),  # stage 4
    (0.933, 0.486, 0.188),  # stage 5 — iEP-like
]
ETIOME_CMAP = LinearSegmentedColormap.from_list('etiome', ETIOME_6)


# ## Load data


from pathlib import Path
SNAPSHOT_PATH = str(Path("../data/Biddy_Patient_Snapshot.h5ad"))
SOURCE_PATH   = str(Path("../data/Biddy_Preprocessed.h5ad"))

adata_syn  = sc.read_h5ad(SNAPSHOT_PATH)
adata_src  = sc.read_h5ad(SOURCE_PATH)

print(f'Snapshot : {adata_syn.n_obs} cells × {adata_syn.n_vars} genes')
print(f'Source   : {adata_src.n_obs} cells × {adata_src.n_vars} genes')


# ## Plot 1 — Sample composition
#
# Stacked horizontal bar showing the fraction of cells from each iEP stage
# within each synthetic sample, sorted by dominant stage.


def plot_sample_composition(adata, figsize=(8, 8)):
    """Stacked horizontal bar of cell-stage proportions per sample."""
    counts = (
        adata.obs
        .groupby(['sample', 'cell_condition'])
        .size()
        .unstack(fill_value=0)
    )
    props = counts.div(counts.sum(axis=1), axis=0)

    # Sort: by dominant stage, then by dominant proportion descending
    dom_stage = props.idxmax(axis=1)
    dom_prop  = props.max(axis=1)
    order = (
        pd.DataFrame({'stage': dom_stage, 'prop': dom_prop})
        .sort_values(['stage', 'prop'], ascending=[True, False])
        .index
    )
    props = props.loc[order]

    # Rename samples by dominant stage
    stage_labels = ['Healthy', 'Early', 'Early', 'Mid', 'Late', 'Severe']
    counters = {n: 1 for n in set(stage_labels)}
    new_names = []
    for s in props.index:
        name = stage_labels[dom_stage[s]]
        new_names.append(f'{name}_{counters[name]:02d}')
        counters[name] += 1
    props.index = new_names
    props = props.iloc[::-1]  # flip so Healthy is at top

    stage_desc = ['Stage 0 (MEF)', 'Stage 1 (Very Early)', 'Stage 2 (Early)',
                  'Stage 3 (Mid)', 'Stage 4 (Late)', 'Stage 5 (iEP)']

    fig, ax = plt.subplots(figsize=figsize)
    left = np.zeros(len(props))
    for stage in range(6):
        if stage in props.columns:
            ax.barh(props.index, props[stage], left=left,
                    label=stage_desc[stage], color=ETIOME_6[stage])
            left += props[stage].values

    ax.set_xlabel('Proportion of cells')
    ax.set_ylabel('Sample')
    ax.set_title('Stage Composition per Sample')
    ax.set_xlim(0, 1)
    ax.tick_params(axis='y', labelsize=5)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    return fig, ax


fig, ax = plot_sample_composition(adata_syn)
fig.savefig(os.path.join(PLOTS_DIR, 'patient_snapshots_sample_composition.png'), dpi=300)
plt.close(fig)


# ## Plot 2 — UMAP of synthetic data
#
# Left: coloured by **sample condition** (0 = healthy, 5 = severe).
# Right: coloured by **cell condition** (per-cell iEP stage).


ETIOME_256 = [
    [0.4157,0.6314,0.6431],[0.4146,0.6369,0.6457],[0.4135,0.6426,0.6482],[0.4125,0.6483,0.6508],
    [0.4114,0.6533,0.6525],[0.4104,0.6559,0.6517],[0.4093,0.6584,0.6507],[0.4083,0.6609,0.6497],
    [0.4073,0.6634,0.6485],[0.4062,0.6659,0.6473],[0.4052,0.6684,0.6459],[0.4042,0.6709,0.6444],
    [0.4032,0.6734,0.6428],[0.4022,0.6759,0.6412],[0.4012,0.6784,0.6394],[0.4002,0.6808,0.6375],
    [0.3992,0.6833,0.6356],[0.3982,0.6858,0.6335],[0.3973,0.6882,0.6313],[0.3963,0.6907,0.629],
    [0.3954,0.6931,0.6267],[0.3944,0.6955,0.6242],[0.3935,0.698,0.6217],[0.3925,0.7004,0.619],
    [0.3916,0.7028,0.6163],[0.3907,0.7052,0.6134],[0.3898,0.7076,0.6105],[0.3888,0.71,0.6075],
    [0.3879,0.7124,0.6043],[0.387,0.7148,0.6011],[0.3861,0.7171,0.5978],[0.3853,0.7195,0.5944],
    [0.3844,0.7219,0.5909],[0.3835,0.7242,0.5874],[0.3826,0.7266,0.5837],[0.3818,0.7289,0.5799],
    [0.3809,0.7313,0.5761],[0.3801,0.7336,0.5721],[0.3792,0.7359,0.5681],[0.3784,0.7382,0.564],
    [0.3776,0.7406,0.5598],[0.3767,0.7429,0.5555],[0.3759,0.7452,0.5512],[0.3751,0.7475,0.5467],
    [0.3743,0.7497,0.5422],[0.3735,0.752,0.5375],[0.3727,0.7543,0.5328],[0.3719,0.7566,0.5281],
    [0.3711,0.7588,0.5232],[0.3704,0.7611,0.5182],[0.3696,0.7633,0.5132],[0.3688,0.7656,0.5081],
    [0.3681,0.7678,0.5029],[0.3673,0.77,0.4976],[0.3666,0.7723,0.4922],[0.3659,0.7745,0.4868],
    [0.3651,0.7767,0.4813],[0.3644,0.7789,0.4757],[0.3637,0.7811,0.47],[0.363,0.7833,0.4643],
    [0.3623,0.7855,0.4585],[0.3616,0.7877,0.4526],[0.3609,0.7898,0.4466],[0.3602,0.792,0.4405],
    [0.3595,0.7942,0.4344],[0.3588,0.7963,0.4282],[0.3582,0.7985,0.4219],[0.3575,0.8006,0.4156],
    [0.3569,0.8028,0.4092],[0.3562,0.8049,0.4027],[0.3556,0.807,0.3961],[0.3549,0.8091,0.3895],
    [0.3543,0.8112,0.3828],[0.3537,0.8133,0.3761],[0.3531,0.8154,0.3692],[0.3525,0.8175,0.3623],
    [0.3519,0.8196,0.3553],[0.3542,0.8217,0.3513],[0.3601,0.8238,0.3507],[0.3662,0.8258,0.3501],
    [0.3722,0.8279,0.3495],[0.3784,0.83,0.3489],[0.3846,0.832,0.3484],[0.391,0.834,0.3478],
    [0.3973,0.8361,0.3472],[0.4038,0.8381,0.3467],[0.4104,0.8401,0.3462],[0.417,0.8421,0.3456],
    [0.4237,0.8442,0.3451],[0.4304,0.8462,0.3446],[0.4373,0.8482,0.3441],[0.4442,0.8501,0.3436],
    [0.4512,0.8521,0.343],[0.4582,0.8541,0.3426],[0.4654,0.8561,0.3421],[0.4726,0.8581,0.3416],
    [0.4798,0.86,0.3411],[0.4872,0.862,0.3406],[0.4946,0.8639,0.3402],[0.5021,0.8659,0.3397],
    [0.5096,0.8678,0.3393],[0.5172,0.8697,0.3388],[0.5249,0.8716,0.3384],[0.5327,0.8736,0.3379],
    [0.5405,0.8755,0.3375],[0.5484,0.8774,0.3371],[0.5563,0.8793,0.3367],[0.5643,0.8812,0.3363],
    [0.5724,0.883,0.3359],[0.5805,0.8849,0.3355],[0.5887,0.8868,0.3351],[0.597,0.8887,0.3347],
    [0.6053,0.8905,0.3343],[0.6137,0.8924,0.3339],[0.6221,0.8942,0.3336],[0.6306,0.8961,0.3332],
    [0.6392,0.8979,0.3329],[0.6478,0.8997,0.3325],[0.6565,0.9015,0.3322],[0.6653,0.9034,0.3318],
    [0.6741,0.9052,0.3315],[0.6829,0.907,0.3312],[0.6918,0.9088,0.3309],[0.7008,0.9106,0.3306],
    [0.7098,0.9123,0.3303],[0.7189,0.9141,0.33],[0.7281,0.9159,0.3297],[0.7373,0.9176,0.3294],
    [0.7412,0.9177,0.3283],[0.7451,0.9178,0.3273],[0.749,0.9179,0.3262],[0.753,0.918,0.3251],
    [0.757,0.9181,0.3241],[0.761,0.9182,0.323],[0.765,0.9183,0.3219],[0.769,0.9184,0.3208],
    [0.773,0.9185,0.3198],[0.7771,0.9186,0.3187],[0.7812,0.9187,0.3176],[0.7853,0.9188,0.3165],
    [0.7894,0.9189,0.3155],[0.7935,0.919,0.3144],[0.7976,0.9191,0.3133],[0.8018,0.9191,0.3122],
    [0.806,0.9192,0.3111],[0.8102,0.9193,0.3101],[0.8144,0.9194,0.309],[0.8186,0.9195,0.3079],
    [0.8229,0.9196,0.3068],[0.8271,0.9197,0.3057],[0.8314,0.9198,0.3047],[0.8357,0.9199,0.3036],
    [0.84,0.9201,0.3025],[0.8444,0.9202,0.3014],[0.8487,0.9203,0.3003],[0.8531,0.9204,0.2992],
    [0.8575,0.9205,0.2982],[0.8619,0.9206,0.2971],[0.8663,0.9207,0.296],[0.8707,0.9208,0.2949],
    [0.8752,0.9209,0.2938],[0.8796,0.921,0.2927],[0.8841,0.9211,0.2916],[0.8886,0.9212,0.2906],
    [0.8931,0.9213,0.2895],[0.8977,0.9214,0.2884],[0.9022,0.9215,0.2873],[0.9068,0.9216,0.2862],
    [0.9114,0.9218,0.2851],[0.916,0.9219,0.284],[0.9206,0.922,0.2829],[0.9221,0.9189,0.2818],
    [0.9222,0.9145,0.2807],[0.9223,0.91,0.2796],[0.9224,0.9055,0.2785],[0.9225,0.901,0.2775],
    [0.9227,0.8965,0.2764],[0.9228,0.892,0.2753],[0.9229,0.8875,0.2742],[0.923,0.8829,0.2731],
    [0.9231,0.8784,0.272],[0.9232,0.8738,0.2709],[0.9234,0.8692,0.2698],[0.9235,0.8645,0.2687],
    [0.9236,0.8599,0.2676],[0.9237,0.8553,0.2665],[0.9238,0.8506,0.2654],[0.924,0.8459,0.2643],
    [0.9241,0.8412,0.2632],[0.9242,0.8365,0.2621],[0.9243,0.8317,0.261],[0.9244,0.827,0.2599],
    [0.9246,0.8222,0.2588],[0.9247,0.8174,0.2577],[0.9248,0.8126,0.2566],[0.9249,0.8078,0.2555],
    [0.9251,0.8029,0.2544],[0.9252,0.7981,0.2532],[0.9253,0.7932,0.2521],[0.9254,0.7883,0.251],
    [0.9256,0.7834,0.2499],[0.9257,0.7785,0.2488],[0.9258,0.7735,0.2477],[0.9259,0.7686,0.2466],
    [0.9261,0.7636,0.2455],[0.9262,0.7586,0.2444],[0.9263,0.7536,0.2433],[0.9265,0.7485,0.2422],
    [0.9266,0.7435,0.241],[0.9267,0.7384,0.2399],[0.9269,0.7334,0.2388],[0.927,0.7283,0.2377],
    [0.9271,0.7231,0.2366],[0.9273,0.718,0.2355],[0.9274,0.7128,0.2344],[0.9275,0.7077,0.2333],
    [0.9277,0.7025,0.2321],[0.9278,0.6973,0.231],[0.9279,0.6921,0.2299],[0.9281,0.6868,0.2288],
    [0.9282,0.6816,0.2277],[0.9283,0.6763,0.2266],[0.9285,0.671,0.2254],[0.9286,0.6657,0.2243],
    [0.9288,0.6604,0.2232],[0.9289,0.655,0.2221],[0.929,0.6496,0.221],[0.9292,0.6443,0.2198],
    [0.9293,0.6389,0.2187],[0.9295,0.6334,0.2176],[0.9296,0.628,0.2165],[0.9298,0.6226,0.2153],
    [0.9299,0.6171,0.2142],[0.93,0.6116,0.2131],[0.9302,0.6061,0.212],[0.9303,0.6006,0.2108],
    [0.9305,0.595,0.2097],[0.9306,0.5895,0.2086],[0.9308,0.5839,0.2075],[0.9309,0.5783,0.2063],
    [0.9311,0.5727,0.2052],[0.9312,0.567,0.2041],[0.9314,0.5614,0.203],[0.9315,0.5557,0.2018],
    [0.9317,0.55,0.2007],[0.9318,0.5443,0.1996],[0.932,0.5386,0.1984],[0.9321,0.5329,0.1973],
    [0.9323,0.5271,0.1962],[0.9324,0.5213,0.195],[0.9326,0.5155,0.1939],[0.9327,0.5097,0.1928],
    [0.9329,0.5039,0.1916],[0.933,0.498,0.1905],[0.9332,0.4922,0.1894],[0.9333,0.4863,0.1882],
]
from matplotlib.colors import LinearSegmentedColormap
ETIOME_CMAP = LinearSegmentedColormap.from_list('etiome', ETIOME_256)


def compute_umap(adata, n_sample=10_000, seed=0):
    """Subsample to n_sample cells, compute HVGs → PCA → neighbours → UMAP."""
    import numpy as np
    if adata.n_obs > n_sample:
        idx = np.random.default_rng(seed).choice(adata.n_obs, n_sample, replace=False)
        adata = adata[idx].copy()
    else:
        adata = adata.copy()
    sc.pp.highly_variable_genes(adata, n_top_genes=2000)
    hvg = adata[:, adata.var['highly_variable']].copy()
    sc.pp.pca(hvg, n_comps=50)
    sc.pp.neighbors(hvg)
    sc.tl.umap(hvg)
    adata.obsm['X_umap'] = hvg.obsm['X_umap']
    return adata


# Synthetic UMAP
adata_syn_umap = compute_umap(adata_syn)

# Add sample-level condition and set etiome palette — matches original exactly
sample_cond = adata_syn_umap.obs.groupby('sample')['condition'].first()
adata_syn_umap.obs['sample_condition'] = (
    adata_syn_umap.obs['sample'].map(sample_cond).astype('category')
)
adata_syn_umap.obs['cell_condition'] = adata_syn_umap.obs['cell_condition'].astype('category')
adata_syn_umap.uns['sample_condition_colors'] = ETIOME_6
adata_syn_umap.uns['cell_condition_colors']   = ETIOME_6

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
sc.pl.umap(adata_syn_umap, color='sample_condition', ax=axes[0], show=False,
           title='Sample Condition (0=Healthy, 5=Severe)')
sc.pl.umap(adata_syn_umap, color='cell_condition',   ax=axes[1], show=False,
           title='Cell Condition (0=MEF, 5=iEP)')
plt.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, 'patient_snapshots_synthetic_umap.png'), dpi=300)
plt.close(fig)


# ## Plot 3 — UMAP of source (original Biddy 2018) data
#
# Left: coloured by **batch** (sequencing day / Step).
# Right: coloured by **iEP score** (continuous, 0 = MEF, 1 = iEP).


adata_src_umap = compute_umap(adata_src)

batch_col = 'batch' if 'batch' in adata_src_umap.obs.columns else 'sample'
n_batches = adata_src_umap.obs[batch_col].nunique()
# Etiome palette sized to number of batches
batch_colors = ETIOME_6[:n_batches] if n_batches <= 6 else [
    ETIOME_256[int(i * 255 / (n_batches - 1))] for i in range(n_batches)
]
adata_src_umap.uns[f'{batch_col}_colors'] = batch_colors

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
sc.pl.umap(adata_src_umap, color=batch_col, ax=axes[0], show=False,
           title=f'Batch ({batch_col})', palette=batch_colors)
sc.pl.umap(adata_src_umap, color='iep_score', ax=axes[1], show=False,
           title='iEP Score (0=MEF, 1=iEP)', cmap=ETIOME_CMAP)
plt.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, 'patient_snapshots_source_umap.png'), dpi=300)
plt.close(fig)


# ## Plot 4 — Cell condition proportions by sample condition


def plot_cell_condition_proportions(obs, figsize=(10, 5)):
    """Stacked bar of cell_condition proportions grouped by sample condition."""
    props = (
        obs.groupby(['condition', 'cell_condition'])
        .size()
        .unstack(fill_value=0)
        .pipe(lambda df: df.div(df.sum(axis=1), axis=0))
    )
    stage_desc = ['Stage 0 (MEF)', 'Stage 1', 'Stage 2',
                  'Stage 3', 'Stage 4', 'Stage 5 (iEP)']
    fig, ax = plt.subplots(figsize=figsize)
    props.plot(kind='bar', stacked=True, ax=ax,
               color=[ETIOME_6[i] for i in range(len(props.columns))],
               legend=True)
    ax.set_xlabel('Sample condition')
    ax.set_ylabel('Proportion')
    ax.set_title('Cell Condition Proportions by Sample Condition')
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles, stage_desc[:len(handles)],
              bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xticks(rotation=0)
    fig.tight_layout()
    return fig, ax


fig, ax = plot_cell_condition_proportions(adata_syn.obs)
fig.savefig(os.path.join(PLOTS_DIR, 'patient_snapshots_condition_proportions.png'), dpi=300)
plt.close(fig)
