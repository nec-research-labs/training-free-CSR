import torch   
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from dataclasses import dataclass
from tqdm import tqdm


@dataclass
class Embedding:
    model_name : str = "Qwen/Qwen3-Embedding-0.6B"
    device : str = "cuda:0"

    def __post_init__(self):
        # if there is an old model, we release it
        try:
            del self.model
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception:
            pass

        # load the model
        self.model = SentenceTransformer(
            self.model_name, 
            trust_remote_code=True, 
            device=self.device
        )

        self.inst_type = "default"
        self.inst_text = ""

    def set_instruction_text(self, inst_type, inst_text):
        self.inst_text = inst_text
        self.inst_type = inst_type
        
        # default : 'Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:'
        try:  # Qwen series
            self.model.prompts["query"] = f"Instruct: {self.inst_text}\nQuery:"
        except Exception:
            pass
        return self

    def encode(self, t, query=False):
        l = t.split(" ")
        n = np.minimum(len(l), 32000)
        try:
            f = True
            while f:
                q_ = " ".join(l[:n])
                try:
                    v = self._encode(q_, query=query)
                    f = False
                except Exception:
                    n -= 100
    
                if n <= 0:
                    break
            if f:
                assert False
        except Exception:
            v = self._encode("random text", query=query)
        return v

    def _encode(self, t, query=False):
        with torch.no_grad():
            t = str(t)
            if "Qwen" in self.model_name or "nemotron-8b" in self.model_name:
                if query:
                    v = self.model.encode(t, prompt_name="query")
                else:
                    v = self.model.encode(t)
            else:
                # change query text
                if query:
                    t = f"{self.inst_text}{t}"
                
                if "intfloat/multilingual-e5-large" in self.model_name:
                    if query:
                        t = "query: " + t
                    else:
                        t = "passage: " + t       

                v = self.model.encode(t)
        return v        

    def load(self, path_emb, dict_text=None, query=True):
        try:
            # if there is the precomputed data, we load it.
            df_emb = pd.read_csv(path_emb, index_col=0)
        except Exception:
            # if there is no such data, we compute it.
            print(path_emb)
            df_emb = pd.DataFrame({
                k : self.encode(v, query=query)
                for k, v in tqdm(dict_text.items())
            }).T    
            df_emb.to_csv(path_emb)
        return df_emb
        