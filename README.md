# genai-glasses-26
Generative AI models (VAE, GAN, Diffusion) on Glasses or No Glasses dataset.
---

## Setup

1. Clone the repo
```bash
git clone https://github.com/cfcmadlad/genai-glasses-26.git
cd genai-glasses-26
```

2. Create and activate environment
```bash
conda create -n genai python=3.10
conda activate genai
```

3. Install dependencies
```bash
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

---

## Data

Dataset: https://www.kaggle.com/datasets/jeffheaton/glasses-or-no-glasses

Get from Google Drive (shared separately):
- `resized64.zip` → unzip and place as `data/resized/`
- `train_corrected.csv` → rename to `train.csv`, place at `data/train.csv`

Also place the original `test.csv` at `data/test.csv` (from Kaggle dataset download).

---

## Update Paths

Open `config.py` and update these to match your machine:

```python
IMG_DIR    = "data/faces-spring-2020/faces-spring-2020/"
RESIZED_DIR = "data/resized/"
TRAIN_CSV  = "data/train.csv"
TEST_CSV   = "data/test.csv"
```

---

## Verify Setup

```bash
python skeleton_test.py
```

Expected output:
```
Total samples: 5000
Batch image shape : torch.Size([64, 3, 64, 64])
Image min/max     : -1.000 / 1.000
Unique labels     : [-1, 0, 1]
Pipeline works :)
```

---

## Files

| File | What it does |
|------|-------------|
| `config.py` | All constants and paths |
| `dataset.py` | PyTorch dataset class with augmentations |
| `preprocess.py` | Resizes all 5000 images to 64x64 |
| `visualize_labels.py` | Shows images with labels overlaid |
| `aug_verification.py` | Verifies augmentations visually |
| `skeleton_test.py` | Verifies full pipeline end to end |


## GAN

### Train baseline
```bash
python train_gan.py --run_name baseline --epochs 50
```

### Train ablations
```bash
python train_gan.py --run_name abl_z64    --z_dim 64         --epochs 50
python train_gan.py --run_name abl_z256   --z_dim 256        --epochs 50
python train_gan.py --run_name abl_d2     --d_steps 2        --epochs 50
python train_gan.py --run_name abl_drop03 --dropout 0.3      --epochs 50
python train_gan.py --run_name abl_smooth --label_smooth 0.1 --epochs 50
```

### Compute FID scores
```bash
python compute_fid_gan.py
```

### Output
Generated images saved to `outputs/gan/<run_name>/`  
Final 6 samples (3 glasses + 3 no-glasses) at `outputs/gan/<run_name>/final_6_samples.png`  
FID scores saved to `outputs/gan/fid_scores_final.csv`

### GAN Ablation Results (50 epochs each)

| Run | Hyperparameter Change | FID ↓ | Observation |
|-----|----------------------|-------|-------------|
| abl_d2 | d_steps 1→2 | **98.70** | Best — stronger D forces G to generate sharper, more realistic faces |
| baseline | z=128, d_steps=1, no dropout, smooth=0 | 152.12 | Strong baseline, well balanced training |
| abl_z256 | z_dim 128→256 | 259.67 | Over-parameterised for 64×64, extra dims add noise |
| abl_drop03 | dropout 0→0.3 | 261.54 | D too weak, generator receives poor feedback |
| abl_z64 | z_dim 128→64 | 276.54 | Insufficient latent space to encode facial variation |
| abl_smooth | label_smooth 0→0.1 | 292.52 | Marginal negative effect, baseline already stable |

### Files

| File | What it does |
|------|-------------|
| `gan.py` | Generator and Discriminator with residual connections |
| `train_gan.py` | Training script with ablation support |
| `compute_fid_gan.py` | Generates 500 diverse images per run and computes FID |