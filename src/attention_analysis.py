import glob as _glob
import pickle

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from tqdm import tqdm


def build_query(content, prefix_text):
    return f"Instruct: {prefix_text}\nQuery: {content}"


def forward(model, tokenizer, text, device):
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=512
    ).to(device)
    with torch.no_grad():
        outputs = model(
            **inputs,
            output_attentions=True,
            output_hidden_states=False,
        )
    emb = F.normalize(
        outputs.last_hidden_state[:, -1, :], p=2, dim=-1
    ).squeeze(0)
    attn_last = outputs.attentions[-1][0].mean(dim=0)[-1, :].cpu()
    return {
        "embedding": emb.cpu(),
        "attn_last": attn_last,
        "n_tokens":  inputs["input_ids"].shape[1],
    }


def get_n_prefix_tokens(tokenizer, prefix_text, content):
    full_text = build_query(content, prefix_text)
    n_full    = tokenizer(full_text, return_tensors="pt")["input_ids"].shape[1]
    n_content = tokenizer(
        str(content), add_special_tokens=False, return_tensors="pt"
    )["input_ids"].shape[1]
    return n_full - n_content


def make_results(model, tokenizer, device, d_prefix, user_texts, desc=""):
    prefix_names = list(d_prefix.keys())
    results = {i: {} for i in range(len(user_texts))}
    for u_idx, text in enumerate(tqdm(user_texts, desc=desc)):
        for p in prefix_names:
            query = build_query(text, d_prefix[p])
            r = forward(model, tokenizer, query, device)
            r["n_prefix"] = get_n_prefix_tokens(tokenizer, d_prefix[p], text)
            results[u_idx][p] = r
    return results


def load_emb_from_csv(csv_path):
    df = pd.read_csv(csv_path, index_col=0)
    return df.index.tolist(), torch.tensor(df.values, dtype=torch.float32)


def load_step1_embeddings(emb_dir, text_type, users, csv_slug, prefix_names):
    emb_by_prefix = {}
    for p in prefix_names:
        csv_path = f"{emb_dir}/{text_type}_{csv_slug}_inst{p}_original.csv"
        csv_users, embs = load_emb_from_csv(csv_path)
        idx_map = {u: i for i, u in enumerate(csv_users)}
        valid = [u for u in users if u in idx_map]
        emb_by_prefix[p] = torch.stack([embs[idx_map[u]] for u in valid])
    return emb_by_prefix, valid


def load_candidate_embeddings(emb_dir, csv_slug, prefix_names):
    cand_by_prefix = {}
    for p in prefix_names:
        csv_path = f"{emb_dir}/candidates_{csv_slug}_inst{p}_original.csv"
        cand_ids, embs = load_emb_from_csv(csv_path)
        cand_by_prefix[p] = {"ids": cand_ids, "embs": embs}
    return cand_by_prefix


def build_d_flag(records):
    def _to_list(v):
        if isinstance(v, str):
            return [x.strip() for x in v.split(",") if x.strip()]
        return list(v)
    d_flag = {}
    for user, r in records.items():
        d_flag[user] = {}
        for item in _to_list(r["candidates_positive"]):
            d_flag[user][item] = 1
        for item in _to_list(r["candidates_negative"]):
            d_flag[user][item] = 0
    return d_flag


def load_job_data(cfg, dir_workspace, version_exp):
    emb_dir = f"{dir_workspace}/embedding_data/{version_exp}/Job"
    base    = f"{dir_workspace}/preprocessed_data/{version_exp}/Job"
    csv_slug     = cfg["csv_slug"]
    prefix_names = list(cfg["d_prefix"].keys())

    with open(f"{base}/text_profile_mid-career_original.pickle", "rb") as f:
        profiles = pickle.load(f)
    with open(f"{base}/records_profile_mid-career.pickle", "rb") as f:
        records = pickle.load(f)

    users      = list(records.keys())
    user_texts = [profiles[u] for u in users]
    d_flag     = build_d_flag(records)
    cand       = load_candidate_embeddings(emb_dir, csv_slug, prefix_names)
    emb_csv, users_valid = load_step1_embeddings(
        emb_dir, "profile_mid-career", users, csv_slug, prefix_names
    )
    print(f"Job: {len(users)} users (CSV valid: {len(users_valid)})")
    return users, user_texts, users_valid, d_flag, cand, emb_csv, emb_dir


def load_ml_data(cfg, dir_workspace, version_exp):
    emb_dir = f"{dir_workspace}/embedding_data/{version_exp}/MovieLens"
    base    = f"{dir_workspace}/preprocessed_data/{version_exp}/MovieLens"
    csv_slug     = cfg["csv_slug"]
    prefix_names = list(cfg["d_prefix"].keys())

    with open(f"{base}/text_profile_original.pickle", "rb") as f:
        profiles = pickle.load(f)
    with open(f"{base}/records_profile.pickle", "rb") as f:
        records = pickle.load(f)

    users      = list(records.keys())
    user_texts = [profiles[u] for u in users]
    d_flag     = build_d_flag(records)
    cand       = load_candidate_embeddings(emb_dir, csv_slug, prefix_names)
    emb_csv, users_valid = load_step1_embeddings(
        emb_dir, "profile", users, csv_slug, prefix_names
    )
    print(f"MovieLens: {len(users)} users (CSV valid: {len(users_valid)})")
    return users, user_texts, users_valid, d_flag, cand, emb_csv, emb_dir


def load_ard_data(cfg, dir_workspace, version_exp, data_dir_name="ARD_CDs_and_Vinyl"):
    emb_dir = f"{dir_workspace}/embedding_data/{version_exp}/{data_dir_name}"
    base    = f"{dir_workspace}/preprocessed_data/{version_exp}/{data_dir_name}"
    csv_slug     = cfg["csv_slug"]
    prefix_names = list(cfg["d_prefix"].keys())

    available = sorted([
        int(p.split("concat_")[1].split("-sample")[0])
        for p in _glob.glob(
            f"{emb_dir}/concat_*_{csv_slug}_instdefault_original.csv"
        )
    ])
    n_sample  = min(available[-1], 5)
    text_type = f"concat_{n_sample}-sample"

    with open(f"{base}/records_{n_sample}-sample.pickle", "rb") as f:
        records = pickle.load(f)
    with open(f"{base}/text_history_original.pickle", "rb") as f:
        th = pickle.load(f)

    users      = list(records.keys())
    user_texts = []
    for u in users:
        history = records[u]["history"]
        text    = "\n".join([
            f"#log {k}\n{th[v]}"
            for k, v in history.items() if v in th
        ])
        user_texts.append(text)

    d_flag = build_d_flag(records)
    cand   = load_candidate_embeddings(emb_dir, csv_slug, prefix_names)
    emb_csv, users_valid = load_step1_embeddings(
        emb_dir, text_type, users, csv_slug, prefix_names
    )
    print(f"{data_dir_name}: {len(users)} users (CSV valid: {len(users_valid)}, {text_type})")
    return users, user_texts, users_valid, d_flag, cand, emb_csv, emb_dir


def compare_output_embeddings(emb_by_prefix, n_users, prefix_names, compare_prefixes, label=""):
    n_p = len(prefix_names)
    sim_matrix = np.zeros((n_p, n_p))
    for i, p1 in enumerate(prefix_names):
        for j, p2 in enumerate(prefix_names):
            sims = (emb_by_prefix[p1] * emb_by_prefix[p2]).sum(dim=1).numpy()
            sim_matrix[i, j] = sims.mean()

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        sim_matrix, annot=True, fmt=".4f",
        xticklabels=prefix_names, yticklabels=prefix_names,
        vmin=0.95, vmax=1.0, cmap="Blues", ax=ax,
    )
    ax.set_title(f"Cosine similarity between prefixes (avg @{n_users} users)\n{label}")
    plt.tight_layout()
    plt.show()

    print(f"Cosine similarity vs prefix=default ({label}, mean ± std):")
    for p in compare_prefixes:
        sims = (emb_by_prefix["default"] * emb_by_prefix[p]).sum(dim=1).numpy()
        print(f"  default vs {p:<10s}: {sims.mean():.6f} ± {sims.std():.6f}")

    return emb_by_prefix, sim_matrix


def analyze_attention_weights(results, n_users, prefix_names, compare_prefixes, label="", u_idx=0):
    from IPython.display import display as _display

    fig, axes = plt.subplots(len(prefix_names), 1, figsize=(13, 3 * len(prefix_names)))
    for ax, p in zip(axes, prefix_names):
        r       = results[u_idx][p]
        n_pref  = r["n_prefix"]
        n_total = r["n_tokens"]
        attn    = r["attn_last"].numpy()
        colors  = ["indianred"] * n_pref + ["skyblue"] * (n_total - n_pref)
        ax.bar(range(n_total), attn, color=colors, width=1.0)
        ax.axvline(x=n_pref - 0.5, color="black", linestyle="--", lw=1.2)
        ax.set_title(
            f"prefix={p}  (prefix={n_pref} tokens / content={n_total - n_pref} tokens)"
        )
        ax.set_xlabel("token position")
        ax.set_ylabel("attention weight")
        ax.set_xlim(-0.5, n_total + 0.5)

    legend_handles = [
        mpatches.Patch(color="indianred", label="prefix (instruction)"),
        mpatches.Patch(color="skyblue",   label="content (user text)"),
    ]
    fig.legend(handles=legend_handles, loc="upper right", fontsize=10)
    fig.suptitle(
        f"Attention at last token ({label}, user {u_idx})\n"
        "Last-token hidden state is used as the embedding vector.",
        fontsize=12, y=1.01,
    )
    plt.tight_layout()
    plt.show()

    attn_summary = pd.DataFrame(index=prefix_names, columns=["prefix_attn", "content_attn"])
    for p in prefix_names:
        p_sums, c_sums = [], []
        for ui in range(n_users):
            r      = results[ui][p]
            attn   = r["attn_last"].numpy()
            n_pref = r["n_prefix"]
            p_sums.append(float(attn[:n_pref].sum()))
            c_sums.append(float(attn[n_pref:].sum()))
        attn_summary.loc[p, "prefix_attn"]  = np.mean(p_sums)
        attn_summary.loc[p, "content_attn"] = np.mean(c_sums)

    attn_summary = attn_summary.astype(float)
    total = attn_summary["prefix_attn"] + attn_summary["content_attn"]
    attn_summary["prefix (%)"]  = (attn_summary["prefix_attn"]  / total * 100).round(1)
    attn_summary["content (%)"] = (attn_summary["content_attn"] / total * 100).round(1)
    _display(attn_summary.round(4))
    return attn_summary


def compute_content_varied_similarity(emb_by_prefix, n_users, prefix_names, label=""):
    result = {}
    for p in prefix_names:
        embs = emb_by_prefix[p][:n_users]
        sim_mat       = (embs @ embs.T).numpy()
        idx           = np.triu_indices(len(embs), k=1)
        pairwise_sims = sim_mat[idx]
        result[p]     = {
            "mean":     float(pairwise_sims.mean()),
            "std":      float(pairwise_sims.std()),
            "all_sims": pairwise_sims,
        }
    print(f"Content-varied similarity ({label}, same prefix, different users):")
    for p in prefix_names:
        r = result[p]
        print(f"  prefix={p:<10s}: {r['mean']:.4f} ± {r['std']:.4f}  (n_pairs={len(r['all_sims'])})")
    return result


def plot_similarity_comparison(prefix_varied_sims, content_varied, compare_prefixes, label=""):
    fig, axes = plt.subplots(
        1, len(compare_prefixes),
        figsize=(5 * len(compare_prefixes), 4), sharey=True,
    )
    if len(compare_prefixes) == 1:
        axes = [axes]

    for ax, p in zip(axes, compare_prefixes):
        cv_sims = content_varied["default"]["all_sims"]
        pv_sims = prefix_varied_sims[p]
        ax.hist(cv_sims, bins=50, alpha=0.5, density=True,
                color="steelblue", label="content-varied (default)")
        ax.hist(pv_sims, bins=50, alpha=0.5, density=True,
                color="indianred", label=f"prefix-varied ({p})")
        ax.axvline(cv_sims.mean(), color="steelblue", linestyle="--", linewidth=1.5)
        ax.axvline(pv_sims.mean(), color="indianred",  linestyle="--", linewidth=1.5)
        ax.set_xlabel("Cosine similarity")
        ax.set_title(f"prefix={p}")
        ax.legend(fontsize=8)

    if compare_prefixes:
        axes[0].set_ylabel("Density")
    fig.suptitle(f"Prefix-varied vs content-varied similarity\n{label}", fontsize=12)
    plt.tight_layout()
    plt.show()


def evaluate_task_performance(
    emb_by_prefix, cand_by_prefix, users, d_flag,
    prefix_names, label="", at_K=5, candidate_size=10,
):
    from IPython.display import display as _display
    from src.scorer import Scorer

    scorer    = Scorer(candidate_size=candidate_size, at_K=at_K)
    perf_rows = []
    for p in prefix_names:
        user_embs = emb_by_prefix[p]
        cand_ids  = cand_by_prefix[p]["ids"]
        cand_embs = cand_by_prefix[p]["embs"]
        sim_matrix = (user_embs @ cand_embs.T).numpy()
        dict_sim   = {
            users[i]: {
                cand_ids[j]: float(sim_matrix[i, j])
                for j in range(len(cand_ids))
            }
            for i in range(len(users))
        }
        df_score   = scorer.compute_all(dict_sim, d_flag)
        mean_score = df_score.mean()
        perf_rows.append({
            "prefix":          p,
            f"Recall@{at_K}":  round(mean_score["recall"], 5),
            f"NDCG@{at_K}":    round(mean_score["ndcg"],   5),
        })

    df_perf = pd.DataFrame(perf_rows).set_index("prefix")
    print(f"\nTask performance ({label}, @{len(users)} users, candidate={candidate_size}, K={at_K})")
    _display(df_perf)
    return df_perf


def compute_pv_cv_emb_sims(emb_by_prefix, compare_prefixes=None, n_cv_pairs=500, seed=42):
    # cosine similarity of output embedding vectors
    # PV (prefix-varied): same user, different prefix vs default; kept per prefix
    # CV (content-varied): different users, same prefix=default
    #
    # PV is deliberately NOT pooled across compare_prefixes: a pooled array would
    # silently change value whenever a caller adds or drops a prefix.
    prefix_names = list(emb_by_prefix.keys())
    if compare_prefixes is None:
        compare_prefixes = [p for p in prefix_names if p != "default"]

    emb_def = emb_by_prefix["default"]
    n_users = emb_def.shape[0]

    pv_by_prefix = {}
    for p in compare_prefixes:
        emb_p = emb_by_prefix[p]
        min_n = min(emb_def.shape[0], emb_p.shape[0])
        pv_by_prefix[p] = F.cosine_similarity(
            emb_def[:min_n], emb_p[:min_n], dim=1
        ).numpy()

    rng = np.random.RandomState(seed)
    max_pairs = n_users * (n_users - 1) // 2
    pairs_set = set()
    while len(pairs_set) < min(n_cv_pairs, max_pairs):
        i, j = rng.choice(n_users, 2, replace=False).tolist()
        if i > j:
            i, j = j, i
        pairs_set.add((i, j))

    cv_sims = []
    for i, j in list(pairs_set):
        cos = F.cosine_similarity(
            emb_def[i].unsqueeze(0),
            emb_def[j].unsqueeze(0),
        ).item()
        cv_sims.append(cos)

    return {"pv_by_prefix": pv_by_prefix, "cv": np.array(cv_sims)}


def compute_pv_cv_cos_sims(cache_path, compare_prefixes=None, n_cv_pairs=500, seed=42):
    # cosine similarity of content-token attention distributions
    # PV (prefix-varied): same user, different prefix
    # CV (content-varied): different users, same prefix=default
    with open(cache_path, "rb") as f:
        results_fwd = pickle.load(f)

    n_users      = len(results_fwd)
    all_prefixes = list(results_fwd[0].keys())
    if compare_prefixes is None:
        compare_prefixes = [p for p in all_prefixes if p != "default"]

    # PV: same user, vary prefix
    pv_sims = []
    for u in range(n_users):
        r0               = results_fwd[u]["default"]
        attn_content_def = r0["attn_last"][r0["n_prefix"]:]
        for p in compare_prefixes:
            rp             = results_fwd[u][p]
            attn_content_p = rp["attn_last"][rp["n_prefix"]:]
            min_len        = min(len(attn_content_def), len(attn_content_p))
            if min_len < 2:
                continue
            cos = F.cosine_similarity(
                attn_content_def[:min_len].unsqueeze(0),
                attn_content_p[:min_len].unsqueeze(0),
            ).item()
            pv_sims.append(cos)

    # CV: different users, fix prefix=default
    rng = np.random.RandomState(seed)
    pairs_set = set()
    while len(pairs_set) < n_cv_pairs:
        i, j = rng.choice(n_users, 2, replace=False).tolist()
        if i > j:
            i, j = j, i
        pairs_set.add((i, j))

    cv_sims = []
    for i, j in list(pairs_set):
        ri      = results_fwd[i]["default"]
        rj      = results_fwd[j]["default"]
        attn_i  = ri["attn_last"][ri["n_prefix"]:]
        attn_j  = rj["attn_last"][rj["n_prefix"]:]
        min_len = min(len(attn_i), len(attn_j))
        if min_len < 2:
            continue
        cos = F.cosine_similarity(
            attn_i[:min_len].unsqueeze(0),
            attn_j[:min_len].unsqueeze(0),
        ).item()
        cv_sims.append(cos)

    return {"pv": np.array(pv_sims), "cv": np.array(cv_sims)}
