pip install transformers einops accelerate

import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import streamlit as st

model = "tiiuae/falcon-7b-instruct"

tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
)

prompt = 'Paraphrase the following sentence delimited by curly brackets into'
style = ' pirate english: '
input_text = '{' + 'It is almost 2pm. Time to go eat.' + '}'

text = prompt + style + input_text

sequences = pipeline(
    text,
    max_length=200,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
)
for seq in sequences:
    st.markdown(f"Result: {seq['generated_text']}")

    
