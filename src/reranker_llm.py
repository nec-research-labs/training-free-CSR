import numpy as np
import pandas as pd
import pickle
import time
from dataclasses import dataclass
from tqdm import tqdm


@dataclass
class LLMReranker:
    candidate_size : int = 10

    def fit(self, llm):
        self.llm = llm

    def _prompt(self, query, d_candidates):
        prompt = f"""# Task
Your task is to recommend exactly 10 items from the provided candidate set, ordered from most to least likely to be preferred by the user.

# Constraints
- Select items only from the provided candidate set; do not invent new items or IDs.
- Do not include any items the user has already interacted with.
- Return only a Python list literal of exactly 10 unique integer item IDs (the keys of the candidate set), ordered by preference, e.g., [8, 4, ...]. Do not output anything else.
- Make the ranking deterministic; if items are equally relevant, break ties by ascending item ID.
- Base your ranking only on the information in this prompt (user history and candidate metadata).

# Data
User Information:
{query}

# Candidate items:
{d_candidates}"""
        return prompt

    def _prepare(self, s_flag, d_documents):
        s = s_flag.copy()
        items_pos = sorted(s[s==1].index.values)
        items_neg = sorted(s[s!=1].index.values)[:self.candidate_size-len(items_pos)]
        
        s = s.loc[items_pos+items_neg]
        s = s.sort_index()
        
        items = s.index.values
        flags = s.values
        d_candidates = {i+1 : d_documents[item] for i, item in enumerate(items)}
        return items, flags, d_candidates

    def _add_score(self, output, items, flags):
        idx = [int(i) - 1 for i in output.split("[")[1].split("]")[0].split(", ")]

        s = pd.Series(0.0, items).copy()
        s.loc[items[idx]] = np.array([1 / (i+1) for i in range(len(idx))])
        df_score = pd.DataFrame([s.values], columns=s.index, index=["score"]).T
        df_score["flag"] = flags
        return df_score

    def infer(self, query, s_flag, d_documents):
        items, flags, d_candidates = self._prepare(s_flag, d_documents)
        prompt = self._prompt(query, d_candidates)
    
        d_res = dict()
        idx = 0
        f = True
        while f:
            try:            
                output, d_log = self.llm(prompt, log=True)  
                d_res[idx] = pd.Series(d_log).drop("input text").to_dict()
                df_score = self._add_score(output, items, flags)
                f = False
            except Exception:
                idx += 1
        
            if idx == 3:
                break
        
        if f:
            output = str([i+1 for i in range(self.candidate_size)])
            df_score = self._add_score(output, items, flags)
        return df_score, d_res

    def load(self, dir_res, d_query, d_flag, d_documents):
        B = 100
        users = list(d_query.keys())
        N = max(1, int(len(users) / B)) if users else 0
        ddict_res = dict()
        for i in range(N):
            path_llmreranker = f"{dir_res}/batch_{i}.pickle"
            try:
                # if there is the precomputed data, we load it.
                with open(path_llmreranker, 'rb') as f:
                    dict_res = pickle.load(f)  
            except Exception:
                # if there is no such data, we compute it.
                print(path_llmreranker)
                users_batch = users[i*B:(i+1)*B]
                dict_res = dict()
                for user in tqdm(users_batch):
                    query = d_query[user]
                    s_flag = pd.Series(d_flag[user])
                    df_score, d_res = self.infer(query, s_flag, d_documents)
                    dict_res[user] = {
                        "score" : df_score.T.to_dict(),
                        "log" : d_res
                    }
                time.sleep(10)
                with open(path_llmreranker, 'wb') as f:
                    pickle.dump(dict_res, f)  
        
            ddict_res.update(dict_res)
        return ddict_res
