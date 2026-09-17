import os
import time
import json
import pandas as pd
from dataclasses import dataclass


@dataclass
class LLM:
    company : str = "OpenAI" 
    model_name : str = "gpt-5.1"  
    api_key : str = None
    api_secret_key : str = None
    api_endpoint : str = None
    api_version : str = "2024-05-01-preview"  
    region_name : str = "us-east-1"  
    temperature : float = 0
    max_tokens : int = 2048  
    path_log : str = None  

    def __post_init__(self):
        if self.company == "Azure_OpenAI":
            from openai import AzureOpenAI
            self.client = AzureOpenAI(
                azure_endpoint = self.api_endpoint,
                api_key = self.api_key,
                api_version = self.api_version,
            )
        elif "Amazon_Bedrock" in self.company:
            try:
                os.environ["AWS_ACCESS_KEY_ID"] = self.api_key
                os.environ["AWS_SECRET_ACCESS_KEY"] = self.api_secret_key
            except Exception:
                pass
            
            import boto3
            self.bedrock_runtime = boto3.client(
                service_name='bedrock-runtime', 
                region_name=self.region_name
            )
        else:
            # self.company == "OpenAI"
            from openai import OpenAI
            self.client = OpenAI(
                api_key=self.api_key
            )

    def __call__(self, prompt, log=False):
        if isinstance(prompt, str):
            messages = [
                {"role" : "system", "content" : "You are an AI assistant that helps people find information."},
                {"role": "user", "content": prompt}
            ]
        else:
            messages = prompt
        
        if "Anthropic" in self.company:
            dict_log = self._anthropic(messages)
        elif "Llama" in self.company:
            dict_log = self._llama(messages)
        elif "Qwen_Amazon" in self.company:
            dict_log = self._qwen_aws(messages)
        else:  # OpenAI and AzureOpenAI
            dict_log = self._openai(messages)

        if self.path_log is not None:
            with open(self.path_log, mode='a') as f:
                f.write(f"{dict_log}\n")

        generated_text = dict_log["output text"]
        if log:
            return generated_text, dict_log
        else:
            return generated_text

    def _openai(self, messages):
        start_time = time.time()
        if ("o1" in self.model_name) or ("o3" in self.model_name) or ("o4" in self.model_name):
            if messages[0]["role"] == "system":
                messages_ = messages[1:]  # remove system text
            else:
                messages_ = messages
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages_
            )
        elif "gpt-5" in self.model_name:
            if messages[0]["role"] == "system":
                messages_ = messages[1:]  # remove system text
            else:
                messages_ = messages

            model_name, reasoning_effort = self.model_name.split("_reasoning_")
            try:
                response = self.client.responses.create(
                    model=model_name,
                    input=messages_,
                    reasoning={"effort": reasoning_effort},
                    temperature=0
                )
            except:                
                response = self.client.responses.create(
                    model=model_name,
                    input=messages_,
                    reasoning={"effort": reasoning_effort},
                    #temperature=0  # reasoning on
                )
        else:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature
            )
        elapsed_time = time.time() - start_time

        if "gpt-5" in self.model_name:
            dict_log = {
                "time" : elapsed_time, 
                "input token" : response.usage.input_tokens,
                "output token" : response.usage.output_tokens + response.usage.output_tokens_details.reasoning_tokens,
                "input text" : messages,
                "output text" : response.output[-1].content[0].text
            }
        else:
            dict_log = {
                "time" : elapsed_time, 
                "input token" : response.usage.prompt_tokens,
                "output token" : response.usage.completion_tokens,
                "input text" : messages,
                "output text" : response.choices[0].message.content
            }
        return dict_log

    def _anthropic(self, messages):
        model_name_ = self.model_name
        if messages[0]["role"] == "system":
            role = messages[0]["content"]
            messages_ = messages[1:]  
            d_ = {
                "anthropic_version": self.api_version,
                "max_tokens" : self.max_tokens,
                "messages": messages_,
                "system" : role
            }
        else:
            messages_ = messages
            d_ = {
                "anthropic_version": self.api_version,
                "max_tokens" : self.max_tokens,
                "messages": messages_
            }

        if "_Thinking" in model_name_:
            model_name_ = model_name_.replace("_Thinking", "")
            d_["max_tokens"] = d_["max_tokens"] + 8000
            d_["thinking"] = {
                "type": "enabled",
                "budget_tokens": 8000
            }
            
        body = json.dumps(d_)  

        start_time = time.time()
        res = self.bedrock_runtime.invoke_model(body=body, modelId=model_name_)
        elapsed_time = time.time() - start_time
        
        response = json.loads(res.get('body').read())
        dict_log = {
            "time" : elapsed_time, 
            "input token" : response["usage"]["input_tokens"],
            "output token" : response["usage"]["output_tokens"],
            "input text" : messages,
            "output text" : [d["text"] for d in response["content"] if "text" in d.keys()][0]
        }
        return dict_log

    def _llama(self, messages):
        formatted_prompt = "<|begin_of_text|>" 
        formatted_prompt += "".join([f"<|start_header_id|>{m['role']}<|end_header_id|>{m['content']}<|eot_id|>" for m in messages]) 
        formatted_prompt += "<|start_header_id|>assistant<|end_header_id|>"
        
        native_request = {
            "prompt": formatted_prompt,
            "max_gen_len": self.max_tokens,
            "temperature": self.temperature,
        }

        start_time = time.time()
        res = self.bedrock_runtime.invoke_model(body=json.dumps(native_request), modelId=self.model_name)
        elapsed_time = time.time() - start_time
        
        response = json.loads(res.get('body').read())
        dict_log = {
            "time" : elapsed_time, 
            "input token" : response["prompt_token_count"],
            "output token" : response["generation_token_count"],
            "input text" : messages,
            "output text" : response["generation"]
        }
        return dict_log

    def _qwen_aws(self, messages):
        body = json.dumps({
            "messages": messages,
            "max_tokens" : self.max_tokens                
        })

        start_time = time.time()
        res = self.bedrock_runtime.invoke_model(body=body, modelId=self.model_name)
        elapsed_time = time.time() - start_time

        response = json.loads(res.get('body').read())
        dict_log = {
            "time" : elapsed_time, 
            "input token" : response["usage"]["prompt_tokens"],
            "output token" : response["usage"]["completion_tokens"],
            "input text" : messages,
            "output text" : response["choices"][0]["message"]["content"]
        }
        return dict_log

    def load_log(self, path_log=None):
        if path_log is None:
            path_log = self.path_log

        def __temporary_parse(t):
            # time, tokens
            t_ = t.split(", 'input text")[0] + "}"
            t_ = t_.replace("'", '"')
            d = json.loads(t_)

            # text
            t_ = t.split("input text': ")[1].replace("\\n", "\n")
            input_text, output_text = t_.split(", 'output text': ")
            d["input text"] = input_text
            d["output text"] = "}".join(output_text.split("}")[:-1])
            return d
            
        with open(path_log, 'r') as f:
            df_log = pd.DataFrame([
                pd.Series(__temporary_parse(t)) 
                for t in f.readlines()
            ])
        return df_log

    def compute_cost(self, s):
        # OpenAI
        ## https://platform.openai.com/docs/pricing
        if "gpt" in self.model_name:
            if "4o-mini" in self.model_name:
                fn_fee = lambda s : (s.iloc[0] * 0.15 + s.iloc[1] * 0.6) / 10**6
            elif "4.1-nano" in self.model_name:
                fn_fee = lambda s : (s.iloc[0] * 0.1 + s.iloc[1] * 0.4) / 10**6
            elif "4.1-mini" in self.model_name:
                fn_fee = lambda s : (s.iloc[0] * 0.4 + s.iloc[1] * 1.6) / 10**6
            elif "4.1" in self.model_name:
                fn_fee = lambda s : (s.iloc[0] * 2 + s.iloc[1] * 8) / 10**6
            elif "5.1" in self.model_name:
                fn_fee = lambda s : (s.iloc[0] * 1.25 + s.iloc[1] * 10) / 10**6
            elif "5.2" in self.model_name:
                fn_fee = lambda s : (s.iloc[0] * 1.75 + s.iloc[1] * 14) / 10**6
            elif "5.4" in self.model_name:
                fn_fee = lambda s : (s.iloc[0] * 2.5 + s.iloc[1] * 15) / 10**6
            elif "4o" in self.model_name:
                fn_fee = lambda s : (s.iloc[0] * 2.5 + s.iloc[1] * 10) / 10**6
                        
        # Anthropic
        elif "claude" in self.model_name:
            if "sonnet-4" in self.model_name:
                fn_fee = lambda s : (s.iloc[0] * 3 + s.iloc[1] * 15) / 10**6

        # Llama
        elif "llama3-3-70b" in self.model_name:
            fn_fee = lambda s : (s.iloc[0] * 0.72 + s.iloc[1] * 0.72) / 10**6      

        # Qwen_AWS
        elif "qwen3-235b-a22b-2507-v1" in self.model_name:
            fn_fee = lambda s : (s.iloc[0] * 0.22 + s.iloc[1] * 0.82) / 10**6   
        
        # GPT-4 
        else:
            print("No matched mode name. We computed the worst case by old gpt-4")
            fn_fee = lambda s : (s.iloc[0] * 60 + s.iloc[1] * 120) / 10**6
        fee = fn_fee(s)
        return fee

    def compute_log(self, path_log=None, df_log=None):
        if path_log is None:
            path_log = self.path_log
        if df_log is None:
            df_log = self.load_log(path_log)

        d_ = {
            "time (sec)" : df_log["time"].sum(), 
            "fee (USD)" : self.compute_cost(df_log[["input token", "output token"]].sum())
        }
        s = pd.Series(d_)
        return s