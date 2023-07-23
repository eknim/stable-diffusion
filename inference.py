from ldm.generate import Generate

# model configuration
gr = Generate(
    "stable-diffusion-1.4",
    embedding_path="./models/embeddings.pt",  # modify the embedding path
)

# model loading
gr.load_model()

# variable initialization
text = "a photo of sculpture Darth Vader riding horse in style of *"

# inference returns a list of tuple
results = gr.prompt2image(prompt=text, outdir="./outputs/", iterations=1, steps=50)

# save the image in outputs folder
for row in results:
    im = row[0]
    seed = row[1]
    # im.save(f"./outputs/image-{seed}.png")
    im.save(f"./outputs/image.png")
