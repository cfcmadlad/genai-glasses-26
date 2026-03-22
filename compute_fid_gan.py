"""
generate_and_fid.py
===================
1. Loads saved checkpoint for each run
2. Generates 500 diverse random images
3. Computes FID against real images

Uses the ORIGINAL generator architecture (no residual connections)
to match the saved checkpoints from the ablation runs.
"""

import os
import shutil
import tempfile
import subprocess
import sys

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

from config import DEVICE, OUTPUT_DIR, MODEL_DIR

# ─────────────────────────────────────────────────────────────────────────────
# Original Generator (matches saved checkpoints — no residual connections)
# ─────────────────────────────────────────────────────────────────────────────
class GeneratorOriginal(nn.Module):
    def __init__(self, z_dim=128, num_classes=2, embed_dim=32, ngf=64):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, embed_dim)
        in_ch = z_dim + embed_dim
        self.net = nn.Sequential(
            nn.ConvTranspose2d(in_ch, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, 3, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z, labels):
        emb = self.label_emb(labels)
        x   = torch.cat([z, emb], dim=1)
        x   = x.unsqueeze(-1).unsqueeze(-1)
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
REAL_IMG_DIR = "data/resized"
NUM_GENERATE = 500
BATCH_SIZE   = 50

RUNS = {
    "baseline":   {"z_dim": 128, "ngf": 64, "embed_dim": 32},
    "abl_z64":    {"z_dim": 64,  "ngf": 64, "embed_dim": 32},
    "abl_z256":   {"z_dim": 256, "ngf": 64, "embed_dim": 32},
    "abl_d2":     {"z_dim": 128, "ngf": 64, "embed_dim": 32},
    "abl_drop03": {"z_dim": 128, "ngf": 64, "embed_dim": 32},
    "abl_smooth": {"z_dim": 128, "ngf": 64, "embed_dim": 32},
}

DESCRIPTIONS = {
    "baseline":   "z=128, d_steps=1, no dropout",
    "abl_z64":    "z_dim reduced to 64",
    "abl_z256":   "z_dim increased to 256",
    "abl_d2":     "d_steps increased to 2",
    "abl_drop03": "dropout=0.3 in discriminator",
    "abl_smooth": "label smoothing=0.1",
}


def find_checkpoint(run_name):
    ckpt_dir = os.path.join(MODEL_DIR, "gan", run_name)
    if not os.path.exists(ckpt_dir):
        return None
    ckpts = sorted([f for f in os.listdir(ckpt_dir) if f.endswith(".pt")])
    if not ckpts:
        return None
    return os.path.join(ckpt_dir, ckpts[-1])


def generate_images(run_name, cfg, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = find_checkpoint(run_name)
    if ckpt_path is None:
        print(f"  [SKIP] No checkpoint found for {run_name}")
        return False
    print(f"  Loading: {ckpt_path}")
    G = GeneratorOriginal(z_dim=cfg["z_dim"], num_classes=2,
                          embed_dim=cfg["embed_dim"], ngf=cfg["ngf"])
    ckpt = torch.load(ckpt_path, map_location="cpu")
    G.load_state_dict(ckpt["G_state"])
    G.eval()
    img_count = 0
    with torch.no_grad():
        while img_count < NUM_GENERATE:
            current_batch = min(BATCH_SIZE, NUM_GENERATE - img_count)
            z      = torch.randn(current_batch, cfg["z_dim"])
            labels = torch.randint(0, 2, (current_batch,))
            fake   = G(z, labels)
            fake   = ((fake + 1) / 2).clamp(0, 1)
            for i in range(current_batch):
                img_pil = transforms.ToPILImage()(fake[i])
                img_pil.save(os.path.join(out_dir, f"gen_{img_count:04d}.png"))
                img_count += 1
    print(f"  Generated {img_count} images to {out_dir}")
    return True


def prepare_real_images(tmp_dir, num=500):
    os.makedirs(tmp_dir, exist_ok=True)
    all_imgs = sorted([f for f in os.listdir(REAL_IMG_DIR) if f.endswith(".png")])
    for f in all_imgs[:num]:
        shutil.copy(os.path.join(REAL_IMG_DIR, f), os.path.join(tmp_dir, f))
    return min(num, len(all_imgs))


def compute_fid(real_dir, fake_dir):
    result = subprocess.run(
        [sys.executable, "-m", "pytorch_fid", real_dir, fake_dir, "--device", "cpu"],
        capture_output=True, text=True
    )
    output = result.stdout + result.stderr
    for line in output.splitlines():
        if "FID" in line:
            try:
                return float(line.split(":")[-1].strip())
            except ValueError:
                continue
    print(f"  [WARN] Could not parse FID:\n{output}")
    return None


def main():
    print("=" * 60)
    print("  Generating diverse images + computing FID")
    print("=" * 60)

    tmp_real = tempfile.mkdtemp(prefix="fid_real_")
    n_real = prepare_real_images(tmp_real, num=500)
    print(f"[INFO] Real images ready: {n_real}")

    results = []
    gen_base = os.path.join(OUTPUT_DIR, "gan_fid_images")

    try:
        for run_name, cfg in RUNS.items():
            print(f"\n{'─'*50}")
            print(f"Run: {run_name}")
            gen_dir = os.path.join(gen_base, run_name)
            ok = generate_images(run_name, cfg, gen_dir)
            if not ok:
                results.append((run_name, None))
                continue
            print(f"  Computing FID...")
            fid = compute_fid(tmp_real, gen_dir)
            if fid is not None:
                print(f"  FID: {fid:.2f}")
            results.append((run_name, fid))
    finally:
        shutil.rmtree(tmp_real)

    print(f"\n{'='*60}")
    print(f"  {'Run':<20} {'FID':>8}  Description")
    print(f"{'─'*60}")
    results_sorted = sorted(results, key=lambda x: x[1] if x[1] is not None else 9999)
    for run, fid in results_sorted:
        fid_str = f"{fid:.2f}" if fid is not None else "N/A"
        print(f"  {run:<20} {fid_str:>8}  {DESCRIPTIONS.get(run, '')}")
    print(f"{'='*60}")

    csv_path = os.path.join(OUTPUT_DIR, "gan", "fid_scores_final.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w") as f:
        f.write("run,fid_score,description\n")
        for run, fid in results:
            fid_str = f"{fid:.4f}" if fid is not None else "N/A"
            f.write(f"{run},{fid_str},{DESCRIPTIONS.get(run,'')}\n")
    print(f"\nSaved to: {csv_path}")


if __name__ == "__main__":
    main()