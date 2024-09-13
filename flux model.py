import torch
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained("black-forest-labs/FLUX.1-dev")

prompt = "A cat holding a sign that says hello world"
image = pipeline(
    prompt,
    height=1024,
    width=1024,
    guidance_scale=3.5,
    num_inference_steps=50,
    max_sequence_length=512,
    generator=torch.Generator("cpu").manual_seed(0)
).images[0]
image.save("flux-dev.png")