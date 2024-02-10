import torch
from transformers import LlamaTokenizer, LlamaForCausalLM

## v2 models
model_path = 'openlm-research/open_llama_13b'

tokenizer = LlamaTokenizer.from_pretrained(model_path,legacy=False)

model = LlamaForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.float16,
).to('cuda')

prompt = 'Q: how many cities are listed in https://deutsche-giganetz.de/ausbau/ ?\nA:'
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to('cuda')

generation_output = model.generate(
    input_ids=input_ids, max_new_tokens=16)

print(tokenizer.decode(generation_output[0]))
    