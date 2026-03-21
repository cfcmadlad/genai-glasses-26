import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision.utils import save_image
import numpy as np
from skimage.metrics import structural_similarity as ssim

from dataset import FacesDataset
from vae import VAE, vae_loss
from config import DEVICE, BATCH_SIZE, NUM_WORKERS, OUTPUT_DIR, MODEL_DIR


LATENT_DIM     = 256      
FILTER_SIZE    = 5        
NUM_LAYERS     = 3        
ACTIVATION     = "leaky_relu"  
DECODER_TYPE   = "interpolation"  
NUM_RES_BLOCKS = 2        
BETA           = 0.5    
LR             = 5e-4     
NUM_EPOCHS     = 100     


def compute_ssim(generated_imgs, real_imgs):
    gen_np  = (generated_imgs.cpu().permute(0,2,3,1).numpy() * 0.5 + 0.5).clip(0, 1)
    real_np = (real_imgs.cpu().permute(0,2,3,1).numpy()      * 0.5 + 0.5).clip(0, 1)
    n       = min(len(gen_np), len(real_np))
    scores  = [ssim(gen_np[i], real_np[i], channel_axis=2, data_range=1.0) for i in range(n)]
    return float(np.mean(scores))


def train():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR,  exist_ok=True)

    full_dataset = FacesDataset(augment=True)
    val_size     = int(len(full_dataset) * 0.1)
    train_size   = len(full_dataset) - val_size

    train_set, val_set = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, drop_last=True)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS)

    model = VAE(
        latent_dim     = LATENT_DIM,
        num_classes    = 2,
        filter_size    = FILTER_SIZE,
        num_layers     = NUM_LAYERS,
        activation     = ACTIVATION,
        decoder_type   = DECODER_TYPE,
        num_res_blocks = NUM_RES_BLOCKS
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Training on {DEVICE}")
    print(f"Train: {len(train_set)} | Val: {len(val_set)}")
    print(f"Model parameters: {total_params:,}\n")

    optimizer = optim.Adam(model.parameters(), lr=LR)
    best_ssim = -1.0

    for epoch in range(1, NUM_EPOCHS + 1):

        model.train()
        total_loss = 0.0
        n_batches  = 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

            labeled   = labels != -1
            unlabeled = ~labeled

            optimizer.zero_grad()

            if labeled.sum() > 0:
                l_imgs, l_labels      = imgs[labeled], labels[labeled]
                recon, mu, logvar     = model(l_imgs, l_labels)
                loss, _, _            = vae_loss(recon, l_imgs, mu, logvar, beta=BETA)
                loss.backward()
                total_loss += loss.item()

            if unlabeled.sum() > 0:
                u_imgs                    = imgs[unlabeled]
                dummy                     = torch.zeros(unlabeled.sum(), dtype=torch.long).to(DEVICE)
                recon_u, mu_u, logvar_u   = model(u_imgs, dummy)
                loss_u, _, _              = vae_loss(recon_u, u_imgs, mu_u, logvar_u, beta=BETA)
                loss_u.backward()
                total_loss += loss_u.item()

            optimizer.step()
            n_batches += 1

        model.eval()
        with torch.no_grad():
            val_imgs, val_labels = next(iter(val_loader))
            val_imgs, val_labels = val_imgs.to(DEVICE), val_labels.to(DEVICE)

            real_labeled  = val_labels != -1
            val_real      = val_imgs[real_labeled]

            n_gen         = min(len(val_real), 32)
            glasses       = model.generate(label=1, n=n_gen//2, device=DEVICE)
            no_glasses    = model.generate(label=0, n=n_gen//2, device=DEVICE)
            generated     = torch.cat([glasses, no_glasses], dim=0)

            val_ssim      = compute_ssim(generated, val_real[:len(generated)])

        print(f"Epoch {epoch:3d} | Loss: {total_loss/n_batches:.4f} | SSIM: {val_ssim:.4f}")

        if val_ssim > best_ssim:
            best_ssim = val_ssim
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, "vae_best.pth"))

        if epoch % 5 == 0 or epoch == 1:
            with torch.no_grad():
                glasses    = model.generate(label=1, n=3, device=DEVICE)
                no_glasses = model.generate(label=0, n=3, device=DEVICE)
                out        = torch.cat([glasses, no_glasses], dim=0)
                out        = (out * 0.5 + 0.5).clamp(0, 1)
                save_image(out, os.path.join(OUTPUT_DIR, f"vae_epoch{epoch}.png"), nrow=3)
                print(f"  Saved outputs/vae_epoch{epoch}.png")

    print(f"\nDone. Best SSIM: {best_ssim:.4f}")
    print(f"Model saved to {MODEL_DIR}/vae_best.pth")


train()