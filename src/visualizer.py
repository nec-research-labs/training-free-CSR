import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines


def set_style(base_fontsize=11):
    plt.rcParams['font.size'] = base_fontsize
    plt.rcParams['axes.grid'] = False
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'


def plot_content_attn_similarity(pv_cv_dict, save_path="figs/content_attn_similarity.pdf",
                                  figsize=(5.5, 3.0),
                                  fontsize_title=13, fontsize_axis=12,
                                  fontsize_tick=11, fontsize_legend=11,
                                  xlim=(0.0, 1.15),
                                  dataset_labels=None,
                                  pv_label="Vary prefix (PV)",
                                  cv_label="Vary content (CV)"):
    set_style()

    datasets = list(pv_cv_dict.keys())
    display_labels = dataset_labels if dataset_labels is not None else datasets
    y = np.arange(len(datasets))
    height = 0.35

    fig, ax = plt.subplots(figsize=figsize)

    pv_means = np.array([pv_cv_dict[ds]["pv"].mean() for ds in datasets])
    pv_stds  = np.array([pv_cv_dict[ds]["pv"].std()  for ds in datasets])
    cv_means = np.array([pv_cv_dict[ds]["cv"].mean() for ds in datasets])
    cv_stds  = np.array([pv_cv_dict[ds]["cv"].std()  for ds in datasets])

    ax.barh(y - height / 2, pv_means, height, xerr=pv_stds,
            color="tab:blue", label=pv_label,
            capsize=3, error_kw={"linewidth": 1.0}, alpha=0.85)
    ax.barh(y + height / 2, cv_means, height, xerr=cv_stds,
            color="tab:orange", label=cv_label, hatch="//",
            edgecolor="white", linewidth=0.5,
            capsize=3, error_kw={"linewidth": 1.0}, alpha=0.85)

    ax.set_xlabel("Cosine similarity", fontsize=fontsize_axis)
    ax.set_yticks(y)
    ax.set_yticklabels(display_labels, fontsize=fontsize_tick)
    ax.tick_params(labelsize=fontsize_tick)
    ax.set_xlim(*xlim)

    ax.legend(fontsize=fontsize_legend, frameon=False, loc="lower right")

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.show()


def plot_content_attn_similarity_multi(pv_cv_dict_by_model, save_path="figs/content_attn_similarity_multi.pdf",
                                        figsize=(7.0, 3.0), model_labels=None,
                                        dataset_labels=None,
                                        fontsize_title=13, fontsize_axis=12,
                                        fontsize_tick=11, fontsize_legend=11,
                                        xlim=(0.0, 1.15)):
    set_style()

    models = list(pv_cv_dict_by_model.keys())
    n_models = len(models)
    if model_labels is None:
        model_labels = models

    fig, axes = plt.subplots(1, n_models, figsize=figsize, sharey=True)
    if n_models == 1:
        axes = [axes]

    height = 0.35
    pv_color, cv_color = "tab:blue", "tab:orange"

    for ax, mkey, mlabel in zip(axes, models, model_labels):
        pv_cv_dict = pv_cv_dict_by_model[mkey]
        datasets = list(pv_cv_dict.keys())
        display_labels = dataset_labels if dataset_labels is not None else datasets
        y = np.arange(len(datasets))

        pv_means = np.array([pv_cv_dict[ds]["pv"].mean() for ds in datasets])
        pv_stds  = np.array([pv_cv_dict[ds]["pv"].std()  for ds in datasets])
        cv_means = np.array([pv_cv_dict[ds]["cv"].mean() for ds in datasets])
        cv_stds  = np.array([pv_cv_dict[ds]["cv"].std()  for ds in datasets])

        ax.barh(y - height / 2, pv_means, height, xerr=pv_stds,
                color=pv_color, capsize=3, error_kw={"linewidth": 1.0}, alpha=0.85)
        ax.barh(y + height / 2, cv_means, height, xerr=cv_stds,
                color=cv_color, hatch="//", edgecolor="white", linewidth=0.5,
                capsize=3, error_kw={"linewidth": 1.0}, alpha=0.85)

        ax.set_title(mlabel, fontsize=fontsize_title, fontweight="bold")
        ax.set_yticks(y)
        ax.set_yticklabels(display_labels, fontsize=fontsize_tick)
        ax.tick_params(labelsize=fontsize_tick, pad=2)
        ax.set_xlim(*xlim)
        ax.set_xlabel("Cosine similarity", fontsize=fontsize_axis)

    legend_handles = [
        mpatches_patch(facecolor=pv_color, label="Vary prefix (PV)"),
        mpatches_patch(facecolor=cv_color, label="Vary content (CV)", hatch="//"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=2,
               fontsize=fontsize_legend, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.13, 1, 1])
    fig.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.show()


def mpatches_patch(facecolor, label, hatch=None):
    from matplotlib.patches import Patch
    return Patch(facecolor=facecolor, label=label, hatch=hatch, edgecolor="white")


def plot_layer_similarity(layer_sims_dict, n_users, save_path="figs/layer_similarity.pdf",
                          figsize=(5.5, 2.5), prefix_style=None,
                          fontsize_title=13, fontsize_axis=12, fontsize_tick=11, fontsize_legend=11):
    set_style()

    if prefix_style is None:
        prefix_style = {
            "product": {"label": "item-oriented", "linestyle": "-", "color": "tab:blue"},
            "user": {"label": "user-oriented", "linestyle": "--", "color": "tab:orange"},
            "ablation": {"label": "reversed", "linestyle": ":", "color": "tab:green"},
        }

    datasets = list(layer_sims_dict.keys())
    fig, axes = plt.subplots(1, len(datasets), figsize=figsize, sharey=True)
    if len(datasets) == 1:
        axes = [axes]

    for ax, ds_name in zip(axes, datasets):
        layer_sims = layer_sims_dict[ds_name]
        for p, style in prefix_style.items():
            if p not in layer_sims:
                continue
            arr = np.array(layer_sims[p][:n_users])
            avg = arr.mean(axis=0)
            std = arr.std(axis=0)
            x = np.arange(len(avg))
            ax.plot(x, avg, label=style["label"], linestyle=style["linestyle"],
                    color=style["color"], linewidth=1.2)
            ax.fill_between(x, avg - std, avg + std, color=style["color"], alpha=0.12)

        ax.set_title(ds_name, fontsize=fontsize_title, fontweight='bold')
        ax.set_xlabel("Layer index", fontsize=fontsize_axis)
        if ax == axes[0]:
            ax.set_ylabel("Cosine similarity to default", fontsize=fontsize_axis)
        ax.tick_params(labelsize=fontsize_tick)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=fontsize_legend,
               frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=[0, 0.13, 1, 1])
    fig.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()


def plot_ner_impact(methods, n0, n1, save_path="figs/ner_impact_summary.pdf",
                    figsize=(5.5, 4.0), color_map=None, type_labels=None, xlim=(-28, 10),
                    p_band=5,
                    fontsize_title=13, fontsize_axis=12, fontsize_tick=11,
                    fontsize_tick_y=10, fontsize_legend=11):
    set_style()

    if color_map is None:
        color_map = {
            "Embedding": "steelblue",
            "Reranker": "teal",
            "LLM": "darkorange",
        }
    if type_labels is None:
        type_labels = (["Embedding"] * 6 + ["Reranker"] * 4 + ["LLM"] * 3)
    colors = [color_map[t] for t in type_labels]

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)
    titles = ["N=0", "N=1"]
    data_list = [n0, n1]
    y_pos = np.arange(len(methods))

    for ax, data, title in zip(axes, data_list, titles):
        ax.barh(y_pos, data, color=colors, height=0.7, edgecolor='white', linewidth=0.3)
        ax.axvline(0, color='black', linewidth=1.2, linestyle='-')
        ax.axvspan(-p_band, p_band, color='gray', alpha=0.12, zorder=0)
        ax.axvline( p_band, color='gray', linewidth=0.7, linestyle='--')
        ax.axvline(-p_band, color='gray', linewidth=0.7, linestyle='--')
        ax.set_title(title, fontsize=fontsize_title, fontweight='bold')
        ax.tick_params(labelsize=fontsize_tick)
        ax.set_xlim(*xlim)

        n_emb = type_labels.count("Embedding")
        n_rnk = type_labels.count("Reranker")
        ax.axhline(n_emb - 0.5, color='gray', linewidth=0.5, linestyle='-')
        ax.axhline(n_emb + n_rnk - 0.5, color='gray', linewidth=0.5, linestyle='-')

    # Right-aligned method names (default HA), matching Multi-shot ICL figure style
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(methods, fontsize=fontsize_tick_y)
    axes[0].invert_yaxis()

    # Single shared x-label under both axes, lifted above the legend
    fig.supxlabel("Avg. relative decrease (%)", fontsize=fontsize_axis, y=0.12)

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color_map[t], label=t) for t in color_map]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=fontsize_legend,
               frameon=False, bbox_to_anchor=(0.5, -0.01))

    fig.tight_layout(rect=[0, 0.18, 1, 1])
    fig.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()


def plot_mrr_heatmap(methods, cols, original, ner, save_path="figs/mrr_heatmap.pdf",
                     figsize=(5.5, 4.0), vmin=0.0, vmax=0.65,
                     fontsize_title=13, fontsize_tick=11, fontsize_annot=12, fontsize_annot_small=10,
                     **kwargs):
    set_style()

    def _compute_ranks(arr):
        ranks = np.zeros_like(arr, dtype=int)
        for j in range(arr.shape[1]):
            order = np.argsort(-arr[:, j])
            for rank, idx in enumerate(order, 1):
                ranks[idx, j] = rank
        return ranks

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    titles = ["Original text", "After NER masking"]
    data_list = [original, ner]

    from matplotlib.colors import TwoSlopeNorm
    vcenter = kwargs.get('vcenter', (vmin + vmax) / 2)
    norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)

    for ax, data, title in zip(axes, data_list, titles):
        ranks = _compute_ranks(data)
        im = ax.imshow(data, cmap='RdYlBu_r', aspect='auto', norm=norm)

        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                r = ranks[i, j]
                fw = 'bold' if r <= 3 else 'normal'
                fs = fontsize_annot if r <= 3 else fontsize_annot_small
                color = 'white' if (data[i, j] > 0.40 or data[i, j] < 0.12) else 'black'
                ax.text(j, i, str(r), ha='center', va='center',
                        fontsize=fs, fontweight=fw, color=color)

        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels(cols, fontsize=fontsize_tick)
        ax.set_title(title, fontsize=fontsize_title, fontweight='bold')
        ax.tick_params(labelsize=fontsize_tick)

        # Draw separator lines between model type groups
        for sep in kwargs.get('separators', [5.5, 9.5]):
            ax.axhline(sep, color='white', linewidth=1.5)

    axes[0].set_yticks(range(len(methods)))
    axes[0].set_yticklabels(methods, fontsize=fontsize_tick)
    axes[1].set_yticks([])

    fig.colorbar(im, ax=axes, shrink=0.6, label='MRR', pad=0.02)
    fig.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()


def plot_concat_vs_separate(df_imp, save_path="figs/concat_vs_separate_ndcg.pdf",
                            figsize=(5.5, 3.5), p_band=5,
                            fontsize_axis=12, fontsize_tick=11, fontsize_legend=11):
    set_style()

    colors = ['steelblue', 'darkorange']
    fig, ax = plt.subplots(figsize=figsize)

    n_models = len(df_imp)
    n_cols = len(df_imp.columns)
    bar_height = 0.35
    y_pos = np.arange(n_models)

    hatches = ['', '//']
    for j, (col, hatch, color) in enumerate(zip(df_imp.columns, hatches, colors)):
        offset = (j - (n_cols - 1) / 2) * bar_height
        vals = df_imp[col].values
        ax.barh(y_pos + offset, vals, height=bar_height,
                color=color, hatch=hatch, edgecolor='white', linewidth=0.5,
                label=col, alpha=0.85)

    ax.axvline(0, color='black', linewidth=1.2, linestyle='-')
    ax.axvline( p_band, color='gray', linewidth=0.7, linestyle='--')
    ax.axvline(-p_band, color='gray', linewidth=0.7, linestyle='--')
    ax.axvspan(-p_band, p_band, color='gray', alpha=0.12, zorder=0)

    n_enc = sum(1 for m in df_imp.index if '[E]' in m)
    if 0 < n_enc < n_models:
        ax.axhline(n_enc - 0.5, color='black', linewidth=0.6, linestyle=':')

    ax.set_xlim(-2 * p_band, 4 * p_band)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_imp.index, fontsize=fontsize_tick)
    ax.invert_yaxis()
    ax.tick_params(axis='x', labelsize=fontsize_tick)
    ax.set_xlabel("Relative improvement over Concat (%)", fontsize=fontsize_axis)

    ax.legend(loc='lower right', fontsize=fontsize_legend, frameon=True, framealpha=0.85)

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()
