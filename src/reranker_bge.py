import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from dataclasses import dataclass

@dataclass
class DenseReranker:
    model_name : str = "BAAI/bge-reranker-v2-m3"
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
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            dtype=torch.bfloat16, 
            trust_remote_code=True,
            device_map=self.device
        ).eval()
    
    def compute(self, query, documents, instruction=None):
        pairs = [[query, doc] for doc in documents]
        with torch.no_grad():
            inputs = self.tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=8192).to(self.model.device)
            logits = self.model(**inputs, return_dict=True).logits.view(-1, ).float()
            scores = torch.sigmoid(logits).tolist()
        return scores
        
        