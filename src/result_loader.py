import copy
import pickle
import numpy as np
import pandas as pd
import scipy as sp
import glob
from tqdm import tqdm
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity

from . import utils


class Loader:
    def __init__(self, scorer, dir_workspace, version_exp, N_icl=[1,3,5,10], flag_replace_NER=False, n_users=None):
        self.scorer = scorer
        self.dir_workspace = dir_workspace
        self.version_exp = version_exp
        self.N_icl = N_icl
        self.flag_replace_NER = flag_replace_NER
        # limit to the first n_users (sorted) per pickle; None means all users
        self.n_users = n_users

    def rename_for_paper(self, model_name):
        m = model_name
        for a in ["embedding", "reranker"]:
            try:
                # when the model is to be loaded from local
                m = m.split(f"{a}_models/")[1]
            except Exception:
                pass

        d = {
            "nvidia/llama-embed-nemotron-8b": "[E] Nemotron (8B)",
            "Qwen/Qwen3-Embedding-8B" : "[E] Qwen3 (8B)",
            "Qwen/Qwen3-Embedding-0.6B" : "[E] Qwen3 (0.6B)",
            "Alibaba-NLP/gte-modernbert-base" : "[E] GTE",
            "BAAI/bge-m3" : "[E] BGE-M3",
            "intfloat/multilingual-e5-large" : "[E] ME5",
            "answerdotai/ModernBERT-large" : "[E] ModernBERT",
            "princeton-nlp/sup-simcse-roberta-large" : "[E] SimCSE",
            "FacebookAI/roberta-large" : "[E] RoBERTa",
            "BM25" : "[R] BM25",
            "BAAI/bge-reranker-v2-m3" : "[R] BGE-M3",
            "Alibaba-NLP/gte-reranker-modernbert-base" : "[R] GTE",
            "Qwen/Qwen3-Reranker-0.6B" : "[R] Qwen3 (0.6B)",
            "Qwen/Qwen3-Reranker-8B" : "[R] Qwen3 (8B)",
            "gpt-4.1-mini-2025-04-14" : "[LLM] GPT-4.1 mini",
            "us.meta.llama3-3-70b-instruct-v1:0": "[LLM] Llama3.3 (70B)",
            "qwen.qwen3-235b-a22b-2507-v1:0" :    "[LLM] Qwen3-235B-A22B",
            "gpt-5.1-2025-11-13_reasoning_none": "[LLM] GPT-5.1",
            "gpt-5.4-2026-03-05_reasoning_none": "[LLM] GPT-5.4",
            "gpt-5.1-2025-11-13_reasoning_medium": "[LLM] GPT-5.1 (medium)",
            "us.anthropic.claude-sonnet-4-5-20250929-v1:0": "[LLM] Sonnet 4.5",
            "MovieLens" : "ML-1M",
            "ARD_CDs_and_Vinyl" : "Music",
            "ARD_Movies_and_TV" : "Movie",
            "ARD_Toys_and_Games" : "Toys",
            "ARD_Sports_and_Outdoors" : "Sports"
        }
        name = d.get(m,m)
        return name

    @staticmethod
    def rename_profile_suffix(suffix):
        d = {
            "mid-career": "with-exp",
            "new-graduate": "no-exp",
        }
        return d.get(suffix, suffix)



    # ======================== embedding ========================
    def _pd_cosine(self, Vu, Vc):
        df_sim = pd.DataFrame(
            cosine_similarity(Vu, Vc),
            index=Vu.index,
            columns=Vc.index
        )
        return df_sim

    def _extract(self, d_emb, dict_data):
        df_emb_candidate = d_emb["candidates"]

        # single text
        d_emb_user = {profile : d_emb[profile] for profile in dict_data["profile"].keys()}
        d_emb_user.update({profile : d_emb[profile] for profile in dict_data["concat"].keys()})

        # multiple texts
        df_emb_history = d_emb["history"]
        d_emb_user.update({
            f"separate_{text_type}" : pd.DataFrame({
                user : df_emb_history.loc[items].mean(axis=0) for user, items in d.items()
            }).T for text_type, d in dict_data["interactions"].items()
        })
        return df_emb_candidate, d_emb_user

    def _load_emb(self, loader, dict_data, model_name_emb):
        text_types = np.concatenate([list(dict_data[a].keys()) for a in ["items", "profile", "concat"]])
        ner = dict_data["ner"]

        dir_emb = f"{loader.dir_workspace}/embedding_data/{loader.version_exp}/{loader.data_name}"
        d_inst = utils.load_inst(model_name_emb, flag_replace_NER=loader.flag_replace_NER)

        d_score = dict()
        for inst_type, inst_text in d_inst.items():
            d_emb = dict()
            for text_type in text_types:
                path_emb = f"{dir_emb}/{text_type}_{utils.rename(model_name_emb)}_inst{inst_type}_{ner}.csv"
                d_emb[text_type] = pd.read_csv(path_emb, index_col=0)

            df_emb_candidate, d_emb_user = self._extract(d_emb, dict_data)

            for text_type, df_emb_user in d_emb_user.items():
                t = text_type.replace("concat_", "").replace("separate_", "")
                d_flag = dict_data["flag"][t]
                # subset users to match flag
                users = list(d_flag.keys())
                df_sim = self._pd_cosine(df_emb_user.loc[users], df_emb_candidate)
                d_score[f"{text_type}:::{inst_type}"] = self.scorer.compute_all(df_sim.T.to_dict(), d_flag)
        return d_score



    # ======================== dense reranker ========================
    def _load_reranker(self, loader, dict_data, model_name_reranker):
        text_types = np.concatenate([list(dict_data[a].keys()) for a in ["profile", "concat", "separate"]])
        ner = dict_data["ner"]

        dir_reranker = f"{loader.dir_workspace}/reranking_data/{loader.version_exp}/{loader.data_name}"
        d_inst = utils.load_inst(model_name_reranker, flag_replace_NER=loader.flag_replace_NER)

        d_score = dict()
        for inst_type, inst_text in d_inst.items():
            for text_type in text_types:
                path_reranker = f"{dir_reranker}/{text_type}_{utils.rename(model_name_reranker)}_inst{inst_type}_{ner}.pickle"
                with open(path_reranker, 'rb') as f:
                    d1 = pickle.load(f)

                # subset users to match flag
                t = text_type.replace("concat_", "").replace("separate_", "")
                subset_users = set(dict_data["flag"][t].keys())
                d1 = {user: d2 for user, d2 in d1.items() if user in subset_users}

                dict_sim = {user : {k : v["score"] for k,v in d2.items()} for user, d2 in d1.items()}
                d_flag = {user : {k : v["flag"] for k,v in d2.items()} for user, d2 in d1.items()}
                d_score[f"{text_type}:::{inst_type}"] = self.scorer.compute_all(dict_sim, d_flag)
        return d_score



    # ======================== LLM reranker ========================
    def _compute_log(self, ddict_res, llm):
        df = pd.concat([pd.DataFrame(d["log"]) for d in ddict_res.values()], axis=1).T
        s_log = llm.compute_log(df_log=df)

        df_score = pd.DataFrame({
            user : self.scorer.compute_user(pd.DataFrame(d["score"]).T)
            for user, d in ddict_res.items()
        }).T
        return df_score, s_log

    def _load_llmreranker(self, loader, dict_data, model_name_llm):
        llm = utils.load_llm(model_name=model_name_llm)
        dir_llmreranker = f"{loader.dir_workspace}/LLMreranking_data/{loader.version_exp}/{loader.data_name}"
        dir_llmreranker += f"/candidate{self.scorer.candidate_size}_atK{self.scorer.at_K}"

        text_types = np.concatenate([list(dict_data[a].keys()) for a in ["profile", "concat"]])
        ner = dict_data["ner"]

        inst_type = "default"
        d_score = dict()
        d_log = dict()
        for text_type in text_types:
            dir_res = f"{dir_llmreranker}/{text_type}_{utils.rename(model_name_llm)}_inst{inst_type}_{ner}"

            # load all pre-generated batch files as-is
            ddict_res = dict()
            for path in glob.glob(f"{dir_res}/*.pickle"):
                with open(path, 'rb') as f:
                    ddict_res.update(pickle.load(f))

            df_score, s_log = self._compute_log(ddict_res, llm)
            d_score[f"{text_type}:::{inst_type}"] = df_score
            d_score[f"{text_type.replace('concat', 'separate')}:::{inst_type}"] = df_score
            d_log[f"{text_type}:::{inst_type}"] = s_log

        return d_score, d_log


    # ======================== load all data ========================
    def load_all_score(self, data_names, model_names_emb, model_names_reranker, model_names_llm):
        ddict_score = defaultdict(dict)
        ddict_log = defaultdict(dict)
        for data_name in data_names:
            print("=="*20, self.rename_for_paper(data_name), "=="*20)
            from .data_loader import Loader as DataLoader
            loader = DataLoader(self.dir_workspace, self.version_exp, data_name, N_icl=self.N_icl, flag_replace_NER=self.flag_replace_NER, n_users=self.n_users)
            dict_data = loader.load_data()

            # load all score data
            dict_score = dict()
            for model_name_emb in tqdm(model_names_emb, desc=f"{'Embedding':10}"):
                dict_score[self.rename_for_paper(model_name_emb)] = self._load_emb(loader, dict_data, model_name_emb)
            for model_name_reranker in tqdm(model_names_reranker, desc=f"{'Reranker':10}"):
                dict_score[self.rename_for_paper(model_name_reranker)] = self._load_reranker(loader, dict_data, model_name_reranker)
            for model_name_llm in tqdm(model_names_llm, desc=f"{'LLM':10}"):
                try:
                    d_score, d_log = self._load_llmreranker(loader, dict_data, model_name_llm)
                    dict_score[self.rename_for_paper(model_name_llm)] = d_score
                    ddict_log[self.rename_for_paper(data_name)][self.rename_for_paper(model_name_llm)] = d_log
                except Exception:
                    pass

            ddict_score[self.rename_for_paper(data_name)] = copy.deepcopy(dict_score)

        # rename keys in dictionaries
        P = ["profile"] + list(np.concatenate([[f"{a}_{n_icl}-sample" for a in ["separate", "concat"]] for n_icl in self.N_icl]))
        dict_data = {p : defaultdict(dict) for p in P}
        for p in P:
            for data_name, dict_score in ddict_score.items():
                for model_name, d_score in dict_score.items():
                    for t, d_ in d_score.items():
                        text_type, inst_type = t.split(":::")
                        if p in text_type:
                            if "profile" in text_type:
                                if data_name == "ML-1M":
                                    a = data_name
                                else:
                                    a = f"{data_name} ({self.rename_profile_suffix(text_type.split('_')[1])})"
                            else:
                                a = data_name
                            dict_data[p][a][f"{model_name}:::{inst_type}"] = pd.DataFrame(d_)

        print("=="*20, "Total computation cost in USD", "=="*20)
        self._calculate_cost(ddict_log)
        return dict_data, ddict_log

    def _calculate_cost(self, ddict_log):
        # Cost for LLM inference
        d_cost = dict()
        for data_name, dict_log in ddict_log.items():
            d_cost[data_name] = {
                m : np.sum([v.loc["fee (USD)"] for v in d_log.values()])
                for m, d_log in dict_log.items()
            }
        df = pd.DataFrame(d_cost)
        df["Sum"] = df.sum(axis=1)
        from IPython.display import display
        display(df.round(1))
        print(df.map(lambda s : f"${s:.1f}$").to_latex(escape=False))


    # ======================== statistics ========================
    def convert_confidence_interval(self, s, confidence=0.9):
        mean = s.mean()
        sem = sp.stats.sem(s)

        n = len(s)
        h = sem * sp.stats.t.ppf((1 + confidence) / 2, n - 1)
        lower = mean - h
        upper = mean + h
        return pd.Series([mean, h], index=['mean', 'width']).to_dict()

    def compute_pvalue(self, df, df_base):
        stat_res, p_value = sp.stats.wilcoxon(
            df.values,
            df_base.values,
            alternative='greater'
        )
        return pd.Series(p_value, index=df.columns).to_dict()

    def add_statistics(self, dict_data, model_name_base):
        # compute statistics
        dict_stat = defaultdict(dict)
        for p, d1 in tqdm(dict_data.items(), desc="Hypothesis Testing"):
            for data_name, d2 in d1.items():
                df_base = d2[model_name_base]

                dd = dict()
                for m, df in d2.items():
                    d_ = df.apply(self.convert_confidence_interval)
                    d_pvalue = self.compute_pvalue(df, df_base)
                    for k,v in d_.items():
                        d_[k]["pvalue"] = d_pvalue[k]
                    dd[m] = d_

                dict_stat[p][data_name] = dd
        return dict_stat

    def reshape_data(self, dict_stat, score_name):
        # reshape
        def _tmp(d1, score_name, axis):
            return pd.DataFrame({
                data_name : {
                    m : d3[score_name][axis]
                    for m, d3 in d2.items()
                } for data_name, d2 in d1.items()
            })

        dict_table = {
            p : {
                axis : _tmp(d1, score_name, axis)
                for axis in ["mean", "width", "pvalue"]
            } for p, d1 in dict_stat.items()
        }
        return dict_table

    def compute_rank(self, df):
        num_cols = df.select_dtypes(include='number').columns
        df_ranks = df[num_cols].rank(axis=0, ascending=False, method='min')
        return df_ranks

    def compute_mrr(self, df_ranks):
        s_mrr = (1 / df_ranks).mean(axis=1)
        return s_mrr



    # ======================== latex ========================
    def add_color(self, df_score):
        def _color(imp):
            dp = {
                0.6: r"\cellcolor{pshigh}",
                0.4: r"\cellcolor{phigh}",
                0.3: r"\cellcolor{pmiddle}",
                0.2: r"\cellcolor{plow}",
                0.1: r"\cellcolor{pslow}"
            }
            dn = {
                -0.05: r"\cellcolor{nslow}",
                -0.1: r"\cellcolor{nlow}",
                -0.2: r"\cellcolor{nmiddle}",
                -0.3: r"\cellcolor{nhigh}",
                -0.4: r"\cellcolor{nshigh}"
            }

            f = ""
            for th in sorted(list(dp.keys())):
                if imp > th*0.5:
                    f = dp[th]
            for th in sorted(list(dn.keys()))[::-1]:
                if imp < th*0.5:
                    f = dn[th]

            return f

        return df_score.map(_color)

    def add_mark(self, df_p, p_value):
        df_ = (df_p.fillna(1) < p_value) * 1
        df_fp = df_.map(lambda s : "^{*}" if s == 1 else "")

        df_ = (df_p.fillna(0) > 1 - p_value) * 1
        df_fn = df_.map(lambda s : "^{\\triangledown}" if s == 1 else "")
        df_f = df_fp + df_fn
        return df_f

    def add_bold(self, df_score):
        def _bold(s):
            idx = s.sort_values(ascending=False).index.values
            i1 = idx[0]
            try:
                i2 = idx[1]
            except Exception:
                i2 = i1
            s = s.map(lambda a : f"{a:.3f}")
            s.loc[i1] = "\\textbf{" + s.loc[i1] + "}"
            s.loc[i2] = "\\underline{" + s.loc[i2] + "}"
            return s

        df_bold = df_score.apply(_bold)
        return df_bold

    def to_tex(self, d, p_value, model_name_base=None):
        df_score = d["mean"]
        df_score = df_score.loc[[i for i in df_score.index if "default" in i]]
        if model_name_base is not None:
            s_base = df_score.loc[model_name_base]
        else:
            s_base = df_score.mean(axis=0)
        df_imp = (df_score / s_base) - 1
        df_color = self.add_color(df_imp)
        df_bold = self.add_bold(df_score)

        df_p = d["pvalue"]
        df_p = df_p.loc[[i for i in df_p.index if "default" in i]]
        df_f = self.add_mark(df_p, p_value)

        df_tex = "$" + df_bold + df_f + "$"
        return df_tex, df_color

    def reindex(self, df):
        df = df.loc[[i for i in df.index if "default" in i]]
        df.index = [i.split(":::")[0] for i in df.index]
        return df

    def to_two_columns(self, df_tex):
        def _tmp(i):
            d = {
                "E" : "E",
                "R" : "R"
            }
            a = i.split("[")[1].split("]")[0]
            a = d.get(a,a)
            b = i.split("]")[1]
            return a + " & " + b
        df_tex.index = [_tmp(i) for i in df_tex.index]
        return df_tex
