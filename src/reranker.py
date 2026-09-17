import numpy as np
import pandas as pd
import pickle
from dataclasses import dataclass
from tqdm import tqdm
from rank_bm25 import BM25Okapi


@dataclass
class Reranker:
    model_name : str = "BAAI/bge-reranker-v2-m3"
    device : str = "cuda:0"
    
    def __post_init__(self):
        if "BM25" in self.model_name:
            self.dense_reranker = None
        else:
            if "Qwen3" in self.model_name:
                from .reranker_qwen3 import DenseReranker
            else:  # bge-m3, gte
                from .reranker_bge import DenseReranker
            self.dense_reranker = DenseReranker(model_name=self.model_name, device=self.device)

        self.inst_type = "default"
        self.inst_text = ""

    def set_instruction_text(self, inst_type, inst_text):
        self.inst_type = inst_type
        self.inst_text = inst_text
        return self
        
    def _compute(self, query, documents):
        def _score(query, documents): 
            if "BM25" in self.model_name:
                _split = lambda t : str(t).lower().split(" ")
                bm25 = BM25Okapi([_split(t) for t in documents])
                scores = bm25.get_scores(_split(query))
                
                # if all scores are 0, perturb with small noise
                n = len(documents)
                if len(set(scores)) < int(n/2):
                    scores += 1e-8*np.random.rand(n)
            else:
                scores = self.dense_reranker.compute(query, documents, instruction=self.inst_text)
            return scores
        
        def _fn(query, documents):
            l = query.split(" ")
            n = np.minimum(len(l), 32000)  # approximate max token

            f = True
            while f:
                q_ = " ".join(l[:n])  # truncated query
                try:
                    scores = _score(q_, documents)
                    f = False
                except Exception:
                    n -= 100

                if n <= 1000:
                    # query is short enough; documents are too long, so score each one separately
                    break

            if f:  # if true computation was not done
                if len(documents) == 1:
                    scores = [np.random.rand()]
                else:
                    assert False 
            return scores
            
        try:
            scores = _fn(query, documents)
        except Exception:
            # when GPU out of memory
            scores = [_fn(query, [doc])[0] for doc in documents]
        
        return scores

    def compute(self, queries, documents):
        if isinstance(queries, str):
            scores = self._compute(queries, documents)
        else:
            S = [self._compute(query, documents) for query in queries]
            scores = pd.DataFrame(S).mean(axis=0).values

        return scores     

    def load(self, path_reranker, d_query, d_flag, d_documents):
        try:
            # if there is the precomputed data, we load it.
            with open(path_reranker, 'rb') as f:
                dict_score = pickle.load(f)
        except Exception:
            # if there is no such data, we compute it.
            print(path_reranker)
            dict_score = dict()
            for user, query in tqdm(d_query.items()):
                d_ = d_flag[user]
                items = list(d_.keys())
                flags = list(d_.values())
                
                documents = [d_documents[item] for item in items]
                scores = self.compute(query, documents)
                
                df_ = pd.DataFrame({"item" : items, "score" : scores, "flag" : flags}).set_index("item")
                dict_score[user] = df_.T.to_dict()
    
            with open(path_reranker, 'wb') as f:
                pickle.dump(dict_score, f)  
        return dict_score
