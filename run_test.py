import torch
from transformers import LlamaTokenizer, LlamaForCausalLM

## v2 models
model_path = 'openlm-research/open_llama_3b_v2'

tokenizer = LlamaTokenizer.from_pretrained(model_path,legacy=False)

model = LlamaForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.float16,
).to('cuda')

prompt = 'Q: What is the largest animal?\nA:'
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to('cuda')

generation_output = model.generate(
    input_ids=input_ids, max_new_tokens=16)

# print(generation_output)

# print(tokenizer.decode(generation_output[0]))

for i in generation_output:
    print(tokenizer.decode(i))
    