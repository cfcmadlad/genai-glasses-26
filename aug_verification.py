import matplotlib.pyplot as plt
import numpy as np

from dataset import FacesDataset


DATASET = FacesDataset(augment=True)
NUM_IMAGES = 9


fig, axes = plt.subplots(3, 3, figsize=(9, 9))

for i, ax in enumerate(axes.flatten()):
    tensor, label = DATASET[i]

    img = tensor.permute(1, 2, 0).numpy()
    img = (img * 0.5 + 0.5).clip(0, 1)

    label_text = "glasses" if label.item() == 1 else "no glasses" if label.item() == 0 else "test"
    ax.imshow(img)
    ax.set_title(label_text)
    ax.axis("off")

plt.tight_layout()
plt.savefig("aug_verification.png")
plt.show()
print("Saved aug_verification.png")