import numpy as np
import pandas as pd


def _reindex(df):
    df = df.loc[[i for i in df.index if "default" in i]].copy()
    df.index = [i.split(":::")[0] for i in df.index]
    return df


def extract_ner_impact(ddict_table, ddict_table_NER, model_names_selected, score_name="ndcg"):
    results = {}
    for label, key in [("N=0", "profile"), ("N=1", "separate_1-sample")]:
        df_orig = _reindex(ddict_table[score_name][key]["mean"])
        df_ner = _reindex(ddict_table_NER[score_name][key]["mean"])
        cols = df_orig.columns.intersection(df_ner.columns)
        impact = (df_ner[cols] - df_orig[cols]) / df_orig[cols] * 100
        results[label] = impact.mean(axis=1)

    methods = [m for m in model_names_selected if m in results["N=0"].index]
    n0 = [float(results["N=0"][m]) for m in methods]
    n1 = [float(results["N=1"][m]) for m in methods]

    type_labels = [
        "Embedding" if m.startswith("[E]") else
        "Reranker" if m.startswith("[R]") else
        "LLM"
        for m in methods
    ]
    return methods, n0, n1, type_labels


def extract_mrr_heatmap(ddict_table, ddict_table_NER, model_names_selected, loader, N_icl):
    conditions = ["profile"] + [f"concat_{n}-sample" for n in N_icl]
    col_labels = ["N=0"] + [f"N={n}" for n in N_icl]
    col_map = dict(zip(conditions, col_labels))

    def _build(ddict_t):
        rows = {}
        for cond in conditions:
            if cond not in ddict_t["ndcg"]:
                continue
            df = _reindex(ddict_t["ndcg"][cond]["mean"])
            s_mrr = loader.compute_mrr(loader.compute_rank(df))
            rows[col_map[cond]] = s_mrr
        df_mrr = pd.DataFrame(rows)
        avail = [m for m in model_names_selected if m in df_mrr.index]
        return df_mrr.reindex(avail)[col_labels]

    df_orig = _build(ddict_table)
    df_ner = _build(ddict_table_NER)

    methods = df_orig.index.tolist()
    return methods, col_labels, df_orig.values, df_ner.values


