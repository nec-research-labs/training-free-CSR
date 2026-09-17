import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from dataclasses import dataclass

@dataclass
class DenseReranker:
    model_name : str = "Qwen/Qwen3-Reranker-8B"
    device : str = "cuda:0"
    
    def __post_init__(self):
        # if there is an old model, we release it
        try:
            del self.model
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception:
            pass

        # to reduce GPU size
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        # load the model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, padding_side='left')
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            trust_remote_code=True,
            device_map=self.device
        ).eval()
        
        self.max_length = 8192
        prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.prefix_tokens = self.tokenizer.encode(prefix, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(suffix, add_special_tokens=False)
        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
        self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")       
        
    def process_inputs(self, pairs):
        inputs = self.tokenizer(
            pairs, padding=False, truncation='longest_first',
            return_attention_mask=False, max_length=self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens)
        )
        for i, ele in enumerate(inputs['input_ids']):
            inputs['input_ids'][i] = self.prefix_tokens + ele + self.suffix_tokens
        inputs = self.tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=self.max_length)
        for key in inputs:
            inputs[key] = inputs[key].to(self.model.device)
        return inputs
    
    @torch.no_grad()
    def compute_logits(self, inputs, **kwargs):
        batch_scores = self.model(**inputs).logits[:, -1, :]
        true_vector = batch_scores[:, self.token_true_id]
        false_vector = batch_scores[:, self.token_false_id]
        batch_scores = torch.stack([false_vector, true_vector], dim=1)
        batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
        scores = batch_scores[:, 1].exp().tolist()
        return scores
    
    def compute(self, query, documents, instruction=None):
        def _format_instruction(instruction, query, doc):
            if instruction is None:
                instruction = 'Given a web search query, retrieve relevant passages that answer the query'
            output = "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(instruction=instruction,query=query, doc=doc)
            return output

        pairs = [_format_instruction(instruction, query, doc) for doc in documents]
        
        # Tokenize the input texts
        inputs = self.process_inputs(pairs)
        scores = self.compute_logits(inputs)
        return scores