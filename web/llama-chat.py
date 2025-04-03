import gradio as gr
from transformers import AutoTokenizer
import transformers
import os

MODEL_ID =  os.getenv("MODEL")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
pipeline = transformers.pipeline(
    "text-generation",
    model=MODEL_ID,
    device_map="auto",
)

def predict(prompt):
  sequences = pipeline(
    prompt,
    do_sample=True,
    temperature=0.2,
    top_p=0.9,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    max_length=100,
  )

  response = ''
  for seq in sequences:
    response += seq['generated_text']

  return response

demo = gr.Interface(
  fn=predict,
  inputs=gr.Textbox(label="Please, write your request here:", placeholder="example: def fibonacci(", lines=5),
  outputs=gr.Textbox(label="Answer (inference):")
)

demo.launch(server_name="0.0.0.0")
