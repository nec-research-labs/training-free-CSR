dir_parent = "./your_directory"
version_exp = "your_version"

import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# set random seed
def set_seed(seed=42):
    import os
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  
    
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
        
    rng = np.random.default_rng(seed)
    return rng


# token counter
def compute_token(text):
    import tiktoken
    encoding = tiktoken.get_encoding("o200k_base")
    return len(encoding.encode(text))


def change_col(t):
    t = str(t)
    def _r(i,a):
        if 'A' <= a <= 'Z':
            a = a.lower()
            if i > 0:
                a = " " + a
        return a
    
    return "".join([_r(i,a) for i, a in enumerate(t)])


import re
import html
def to_text(original_text):
    text = re.sub(r'<[^>]+>', '\n', original_text) 
    text = html.unescape(text)
    
    text = text.replace('\\r', '\n').replace('\r', '\n')
    text = text.replace("u\'", "").replace("\'", "")
    lines = [line.strip() for line in text.split('\n')]
    lines = [line for line in lines if line] 
    
    cleaned_text = '\n'.join(lines)
    return cleaned_text


def load_inst(model_name, flag_replace_NER=False):
    d_inst = {
        "default": ""
    }
    if flag_replace_NER is not True:
        # Qwen3-8B: embedding and reranker
        if "Qwen3" in model_name and "8B" in model_name:
            d_inst = {
                "default": "Given a web search query, retrieve relevant passages that answer the query", 
                "product": "Given user information, retrieve relevant items that are similar to the item", 
                "user": "Given user information, retrieve relevant items that the target user would like",
                "ablation" : 'Given a web search query, retrieve "irrelevant" passages that answer the query'
            }    
        if "nemotron-8b" in model_name:
            d_inst = {
                "default": "Given a question, retrieve passages that answer the question", 
                "product": "Given user information, retrieve relevant items that are similar to the item", 
                "user": "Given user information, retrieve relevant items that the target user would like",
                "ablation" : 'Given a question, retrieve passages that DO NOT answer the question'
            }    
    return d_inst


def rename(model_name):
    m = model_name
    for a in ["embedding", "reranker"]:
        try:
            # when the model is to be loaded from local
            m = m.split(f"{a}_models/")[1]
        except Exception:
            pass
    name = m.replace('/','_').replace('.', '_').replace(":", "_")
    return name
   

def load_llm(model_name="gpt-5.1"):
    from src.llm import LLM
    if "llama" in model_name:
        llm = LLM(
            company="Llama_Amazon_Bedrock",
            model_name=model_name,
            region_name="us-east-1", 
            api_key=os.environ.get("AWS_ACCESS_KEY"),
            api_secret_key=os.environ.get("AWS_SECRET_ACCESS_KEY")
        )         
    elif "claude" in model_name:      
        llm = LLM(
            company="Anthropic_Amazon_Bedrock",
            model_name=model_name,
            region_name="us-east-1", 
            api_key=os.environ.get("AWS_ACCESS_KEY"),
            api_secret_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
            api_version="bedrock-2023-05-31"
        )
    elif "qwen.qwen3" in model_name:
        llm = LLM(
            company="Qwen_Amazon_Bedrock",
            model_name=model_name,
            region_name="us-west-2", 
            api_key=os.environ.get("AWS_ACCESS_KEY"),
            api_secret_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
            api_version="bedrock-2023-05-31"
        )        
    else:  # OpenAI
        llm = LLM(
            company="OpenAI",
            model_name=model_name,
            api_key=os.environ.get("OPENAI_API_KEY") 
        )
    return llm