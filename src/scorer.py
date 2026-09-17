import copy
import numpy as np
import pandas as pd
from tqdm import tqdm
from dataclasses import dataclass, field
from sklearn.metrics import ndcg_score


@dataclass
class Scorer:
    candidate_size : int = 50
    at_K : int = 10
    def __post_init__(self):
        if self.candidate_size == 10 and self.at_K == 5:
            d_ = {"recall" : 0.5, "ndcg" : 0.358472}
        elif self.candidate_size == 50 and self.at_K == 10:
            d_ = {"recall" : 0.2, "ndcg" : 0.11403}
        else:
            d = dict()
            for idx in tqdm(range(10000)):
                df_score = pd.DataFrame({
                    "flag" : [1]*2 + [0]*(self.candidate_size-2),
                    "score" : np.random.rand(self.candidate_size)
                })
                d[idx] = self.compute_user(df_score)
            d_ = pd.DataFrame(d).mean(axis=1).to_dict()
        self.d_random = copy.deepcopy(d_)
        
    def compute_user(self, df_score):
        df_ = df_score.sort_values(by="score", ascending=False) 
        s = df_["flag"]
        d_ = {
            "recall" : s.iloc[:self.at_K].sum() / s.sum(),
            "ndcg" : ndcg_score(
                np.asarray([df_["flag"].values]), 
                np.asarray([df_["score"].values]),
                k=self.at_K
            )
        }
        return d_

    def compute_all(self, dict_sim, d_flag):
        d_score = dict()
        for user, d_sim in dict_sim.items():
            s = pd.Series(d_flag[user])
            items_pos = sorted(s[s==1].index.values)
            items_neg = sorted(s[s!=1].index.values)[:self.candidate_size-len(items_pos)]
            df_score = pd.DataFrame({
                "score" : pd.Series(d_sim).loc[items_pos+items_neg].values,
                "flag" : [1]*len(items_pos) + [0]*len(items_neg)
            })
            d_score[user] = self.compute_user(df_score)
        df_score = pd.DataFrame(d_score).T
        return df_score
