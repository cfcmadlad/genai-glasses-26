
import os
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from config import (DEVICE, BATCH_SIZE, NUM_EPOCHS, LR,
                    NUM_WORKERS, NUM_CLASSES, IMG_SIZE, MODEL_DIR, OUTPUT_DIR)
from dataset import FacesDataset
from diffusion import UNet, GaussianDiffusion


def parse_args():
    p = argparse.ArgumentParser(description="Train DDPM on glasses dataset")

    p.add_argument("--dim", type=int, default=64, help="UNet base channel width")
    p.add_argument("--dim_mults", type=int, nargs="+", default=[1, 2, 4, 8],
                   help="Channel multipliers per level")
    p.add_argument("--dropout", type=float, default=0.1, help="Dropout in ResNet blocks")

    p.add_argument("--timesteps", type=int, default=1000, help="Number of diffusion steps T")
    p.add_argument("--beta_schedule", type=str, default="linear",
                   choices=["linear", "cosine"], help="Beta schedule type")
    p.add_argument("--beta_start", type=float, default=1e-4)
    p.add_argument("--beta_end", type=float, default=0.02)

    p.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    p.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--grad_clip", type=float, default=1.0)

    p.add_argument("--sample_every", type=int, default=5,
                   help="Generate samples every N epochs")
    p.add_argument("--num_samples", type=int, default=6,
                   help="Number of images to generate per class when sampling")
    return p.parse_args()


def train(args):
    print(f"Device: {DEVICE}")
    print(f"Config: T={args.timesteps}, schedule={args.beta_schedule}, "
          f"lr={args.lr}, batch={args.batch_size}, dim={args.dim}, "
          f"mults={args.dim_mults}, dropout={args.dropout}")

    save_dir = os.path.join(OUTPUT_DIR, "diffusion")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    dataset = FacesDataset(augment=True)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=NUM_WORKERS, drop_last=True)
    print(f"Dataset size: {len(dataset)}")

    unet = UNet(
        dim=args.dim,
        num_classes=NUM_CLASSES,
        dim_mults=tuple(args.dim_mults),
        channels=3,
        dropout=args.dropout,
    )
    diffusion = GaussianDiffusion(
        model=unet,
        timesteps=args.timesteps,
        beta_schedule=args.beta_schedule,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
    ).to(DEVICE)

    num_params = sum(p.numel() for p in diffusion.parameters() if p.requires_grad)
    print(f"Trainable parameters: {num_params:,}")

    optimiser = torch.optim.Adam(diffusion.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        diffusion.train()
        epoch_loss = 0.0

        pbar = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}")
        for images, labels in pbar:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            loss = diffusion(images, labels)

            optimiser.zero_grad()
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(diffusion.parameters(), args.grad_clip)
            optimiser.step()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = epoch_loss / len(loader)
        print(f"Epoch {epoch} — avg loss: {avg_loss:.4f}")

        if epoch % args.sample_every == 0 or epoch == args.epochs:
            generate_samples(diffusion, epoch, save_dir, args.num_samples)

        if epoch % 10 == 0 or epoch == args.epochs:
            ckpt_path = os.path.join(MODEL_DIR, f"ddpm_epoch{epoch}.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": diffusion.state_dict(),
                "optimiser_state_dict": optimiser.state_dict(),
                "loss": avg_loss,
                "args": vars(args),
            }, ckpt_path)
            print(f"Checkpoint saved: {ckpt_path}")


@torch.no_grad()
def generate_samples(diffusion, epoch, save_dir, n_per_class=3):
    diffusion.eval()

    all_images = []
    for cls in range(NUM_CLASSES):
        labels = torch.full((n_per_class,), cls, dtype=torch.long)
        imgs = diffusion.sample(labels, image_size=IMG_SIZE)
        all_images.append(imgs)

    grid = torch.cat(all_images, dim=0)  # (2*n, C, H, W)
    path = os.path.join(save_dir, f"samples_epoch{epoch}.png")
    save_image(grid, path, nrow=n_per_class, padding=2)
    print(f"Samples saved: {path}")


@torch.no_grad()
def generate_final(ckpt_path, n_per_class=3):
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    hparams = ckpt["args"]

    unet = UNet(
        dim=hparams["dim"],
        num_classes=NUM_CLASSES,
        dim_mults=tuple(hparams["dim_mults"]),
        channels=3,
        dropout=hparams.get("dropout", 0.1),
    )
    diffusion = GaussianDiffusion(
        model=unet,
        timesteps=hparams["timesteps"],
        beta_schedule=hparams["beta_schedule"],
        beta_start=hparams.get("beta_start", 1e-4),
        beta_end=hparams.get("beta_end", 0.02),
    ).to(DEVICE)

    diffusion.load_state_dict(ckpt["model_state_dict"])
    diffusion.eval()

    save_dir = os.path.join(OUTPUT_DIR, "diffusion", "final")
    os.makedirs(save_dir, exist_ok=True)

    for cls in range(NUM_CLASSES):
        label_name = "glasses" if cls == 1 else "no_glasses"
        labels = torch.full((n_per_class,), cls, dtype=torch.long)
        imgs = diffusion.sample(labels, image_size=IMG_SIZE)
        for i in range(n_per_class):
            path = os.path.join(save_dir, f"{label_name}_{i+1}.png")
            save_image(imgs[i], path)
        print(f"Saved {n_per_class} {label_name} images to {save_dir}")


if __name__ == "__main__":
    args = parse_args()
    train(args)
