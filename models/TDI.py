import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from torch.nn.parallel import DataParallel
from models.model_info import LLM_CONFIGS

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.mn = configs.mn
        if self.mn!='openelm':
            mn_ckpt =LLM_CONFIGS[self.mn]['checkpoint']
            tn=mn_ckpt
        else:
            mn_ckpt=LLM_CONFIGS[self.mn]['checkpoint']
            tn='meta-llama/Llama-2-7b-hf'

        self.llm = AutoModelForCausalLM.from_pretrained(
            mn_ckpt,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            cache_dir=configs.cache_dir
        )
        self.llm = self.llm.cuda()
        self.llm_tokenizer = AutoTokenizer.from_pretrained(tn)
        self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token
        self.vocab_size = self.llm_tokenizer.vocab_size
    
        for name, param in self.llm.named_parameters():
            param.requires_grad = False

    def tokenizer(self, x):
        output = self.llm_tokenizer(x, padding=True, return_tensors="pt")['input_ids']
        result = self.llm.get_input_embeddings()(output.cuda())
        return result   
    
    def forecast(self, x_mark_enc):        
        # x_mark_enc: [bs x T x hidden_dim_of_llama]
        x_mark_enc = self.tokenizer(x_mark_enc)
        if self.mn=='openelm' or self.mn=='gpt':
            text_outputs = self.llm.transformer(inputs_embeds=x_mark_enc)[0]
        else:
            text_outputs = self.llm.model(inputs_embeds=x_mark_enc)[0]
        text_outputs = text_outputs[:, -1, :]
        return text_outputs
    
    def forward(self, x_mark_enc):
        return self.forecast(x_mark_enc)