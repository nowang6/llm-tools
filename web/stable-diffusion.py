import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import gradio as gr
import os

model_id = "runwayml/stable-diffusion-v1-5"

# Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")


def get_image(prompt):
  image = pipe(prompt).images[0]
  image.save("astronaut_rides_horse.png")
  return image

demo = gr.Interface(fn=get_image,
                    inputs = [gr.Textbox(label="Enter the Prompt")],
                    outputs = gr.Image(type='pil'), title = "hello", description = "iamge")

demo.launch(server_name="0.0.0.0")