import os
import csv
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, random_split
from torchvision.utils import save_image
import numpy as np
from skimage.metrics import structural_similarity as ssim

from dataset import FacesDataset
from vae import VAE, vae_loss, PerceptualLoss
from config import DEVICE, BATCH_SIZE, NUM_WORKERS, OUTPUT_DIR, MODEL_DIR


def compute_ssim(recon_imgs, real_imgs):
    gen_np  = (recon_imgs.cpu().permute(0,2,3,1).numpy() * 0.5 + 0.5).clip(0, 1)
    real_np = (real_imgs.cpu().permute(0,2,3,1).numpy()  * 0.5 + 0.5).clip(0, 1)
    n       = min(len(gen_np), len(real_np))
    scores  = [ssim(gen_np[i], real_np[i], channel_axis=2, data_range=1.0) for i in range(n)]
    return float(np.mean(scores))


def get_loaders():
    train_dataset = FacesDataset(augment=True)
    val_dataset   = FacesDataset(augment=False)
    val_size      = int(len(train_dataset) * 0.1)
    train_size    = len(train_dataset) - val_size
    indices       = list(range(len(train_dataset)))
    train_idx, val_idx = random_split(indices, [train_size, val_size],
                                      generator=torch.Generator().manual_seed(42))
    train_set = Subset(train_dataset, list(train_idx))
    val_set   = Subset(val_dataset,   list(val_idx))
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, drop_last=True)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS)
    return train_loader, val_loader


def run(run_name, latent_dim=256, filter_size=3, num_layers=3,
        activation="elu", decoder_type="interpolation",
        num_res_blocks=2, beta=0.2, lr=1e-4,
        perc_weight=0.05, num_epochs=50):

    run_out = os.path.join(OUTPUT_DIR, "ablations", run_name)
    run_mod = os.path.join(MODEL_DIR,  "ablations", run_name)
    os.makedirs(run_out, exist_ok=True)
    os.makedirs(run_mod, exist_ok=True)

    train_loader, val_loader = get_loaders()

    model = VAE(latent_dim=latent_dim, num_classes=2, filter_size=filter_size,
                num_layers=num_layers, activation=activation,
                decoder_type=decoder_type, num_res_blocks=num_res_blocks).to(DEVICE)
    perc_fn   = PerceptualLoss().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_ssim = -1.0

    print(f"\n{'='*50}\nRun: {run_name} | Params: {sum(p.numel() for p in model.parameters()):,}\n{'='*50}")

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss, n_batches = 0.0, 0

        for imgs, labels in train_loader:
            imgs, labels     = imgs.to(DEVICE), labels.to(DEVICE)
            labeled          = labels != -1
            unlabeled        = ~labeled
            optimizer.zero_grad()

            if labeled.sum() > 0:
                l_imgs, l_labels  = imgs[labeled], labels[labeled]
                recon, mu, logvar = model(l_imgs, l_labels)
                loss, _, _        = vae_loss(recon, l_imgs, mu, logvar, beta=beta)
                total_l           = loss + perc_fn(recon, l_imgs) * perc_weight
                total_l.backward()
                total_loss       += total_l.item()

            if unlabeled.sum() > 0:
                u_imgs                  = imgs[unlabeled]
                dummy                   = torch.zeros(unlabeled.sum(), dtype=torch.long).to(DEVICE)
                recon_u, mu_u, logvar_u = model(u_imgs, dummy)
                loss_u, _, _            = vae_loss(recon_u, u_imgs, mu_u, logvar_u, beta=beta)
                total_u                 = loss_u + perc_fn(recon_u, u_imgs) * perc_weight
                total_u.backward()
                total_loss             += total_u.item()

            optimizer.step()
            n_batches += 1

        model.eval()
        with torch.no_grad():
            all_recon, all_real = [], []
            for val_imgs, val_labels in val_loader:
                val_imgs, val_labels = val_imgs.to(DEVICE), val_labels.to(DEVICE)
                real_labeled         = val_labels != -1
                val_real             = val_imgs[real_labeled]
                val_real_labels      = val_labels[real_labeled]
                if len(val_real) == 0:
                    continue
                recon, _, _ = model(val_real, val_real_labels)
                all_recon.append(recon)
                all_real.append(val_real)

            all_recon = torch.cat(all_recon)
            all_real  = torch.cat(all_real)
            val_ssim  = compute_ssim(all_recon, all_real)

        print(f"Epoch {epoch:3d} | Loss: {total_loss/n_batches:.4f} | SSIM: {val_ssim:.4f}")

        if val_ssim > best_ssim:
            best_ssim = val_ssim
            torch.save(model.state_dict(), os.path.join(run_mod, "best.pth"))

        if epoch % 10 == 0 or epoch == num_epochs:
            with torch.no_grad():
                glasses    = model.generate(label=0, n=3, device=DEVICE)
                no_glasses = model.generate(label=1, n=3, device=DEVICE)
                out = (torch.cat([glasses, no_glasses]) * 0.5 + 0.5).clamp(0, 1)
                save_image(out, os.path.join(run_out, f"epoch{epoch}.png"), nrow=3)

    print(f"\n[{run_name}] Best SSIM: {best_ssim:.4f}")
    return best_ssim


results = {}

results["baseline"]   = run("baseline")
results["latent_64"]  = run("latent_64",  latent_dim=64)
results["latent_512"] = run("latent_512", latent_dim=512)
results["filter_5"]   = run("filter_5",   filter_size=5)
results["filter_7"]   = run("filter_7",   filter_size=7)
results["deconv"]     = run("deconv",     decoder_type="deconv")
results["act_relu"]   = run("act_relu",   activation="relu")
results["act_leaky"]  = run("act_leaky",  activation="leaky_relu")
results["layers_2"]   = run("layers_2",   num_layers=2)
results["layers_4"]   = run("layers_4",   num_layers=4)
results["no_res"]     = run("no_res",     num_res_blocks=0)
results["res_1"]      = run("res_1",      num_res_blocks=1)

print(f"\n{'='*40}")
print(f"{'Run':<20} {'Best SSIM':>10}")
print("-" * 32)
for name, score in results.items():
    print(f"{name:<20} {score:>10.4f}")

os.makedirs(os.path.join(OUTPUT_DIR, "ablations"), exist_ok=True)
with open(os.path.join(OUTPUT_DIR, "ablations", "ssim_results.csv"), "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["run", "ssim"])
    for name, score in results.items():
        writer.writerow([name, f"{score:.4f}"])

print("\nSaved to outputs/ablations/ssim_results.csv")