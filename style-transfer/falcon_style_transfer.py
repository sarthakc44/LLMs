# -*- coding: utf-8 -*-
"""falcon-style-transfer [LLM HF Gradio]

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1NBvlDULrAADXYZSwUnMOhYohpBRlzXtA

## Falcon-7b-instruct
"""

!pip install transformers einops accelerate

import locale
print(locale.getpreferredencoding())

def getpreferredencoding(do_setlocale = True):
    return "UTF-8"
    # needed for gradio
locale.getpreferredencoding = getpreferredencoding

print(locale.getpreferredencoding())

!pip install gradio

from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers, torch, gradio as gr

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
style = ' exaggerated victorian english: '
input_text = '{' + 'Almost lunchtime. Time to eat!' + '}'

text_prompt = prompt + style + input_text

def llm(input_text, style):
  prompt = 'Paraphrase and change the style of the following sentence delimited by curly brackets into an exaggerated '
  style = style + ' accent: '

  text_prompt = prompt + style + '{' + input_text + '}'

  sequences = pipeline(
      text_prompt,
      max_length=256,
      do_sample=True,
      num_return_sequences=1,
      eos_token_id=tokenizer.eos_token_id,
      return_full_text=False
  )

  output_text = ''
  for seq in sequences:
      output_text = output_text + seq['generated_text']

  return output_text

#for seq in sequences:
 #   print(f"Result: {seq['generated_text']}")

title = "Change your Speaking Style!"
description = """
Write something, select an accent, and change the style of your text in seconds!

Didn't like the response? Just click on submit again!
"""

article = "This demo uses the [Falcon-7b-Instruct Model](https://huggingface.co/tiiuae/falcon-7b-instruct) and is purely for recreational purposes. View the source code on the [github repo.](https://github.com/sarthakc44/LLMs/tree/main/style-transfer)"


textbox = gr.Textbox(label="Type a few sentences below:", placeholder="Almost lunchtime. Time to eat!", lines=3)
radio = gr.Radio(["Crazy Pirate", "Formal Victorian", "Hillbilly Southern", "Casual Talkative", "Flowery Poetic",], label="Choose your accent!")

demo = gr.Interface(
  fn=llm,
  inputs=[textbox,
          radio,],
  outputs="text",
  title=title,
  description=description,
  article=article,
).launch()