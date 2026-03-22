import torch
import clip
from PIL import Image
import pandas as pd
import os

# Install first if needed: pip install clip git+https://github.com/openai/CLIP.git

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Text prompts
text = clip.tokenize(["a person wearing glasses", "a person not wearing glasses"]).to(device)

test_df = pd.read_csv("data/test.csv")
results = []

for _, row in test_df.iterrows():
    filename = f"face-{int(row['id'])}.png"
    img_path = os.path.join("data/resized", filename)
    
    if not os.path.exists(img_path):
        continue
    
    image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits_per_image, _ = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]
    
    # 1 = glasses, 0 = no glasses
    label = 1 if probs[0] > probs[1] else 0
    results.append({"id": row["id"], "glasses": label})

labeled_df = pd.DataFrame(results)
labeled_df.to_csv("data/test_labeled.csv", index=False)
print(f"Labeled {len(labeled_df)} images")
print(labeled_df["glasses"].value_counts())