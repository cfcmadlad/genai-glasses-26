import os
import csv
import torch
from torch.utils.data import DataLoader, Subset, random_split

from dataset import FacesDataset
from vae import VAE
from config import DEVICE, BATCH_SIZE, NUM_WORKERS, OUTPUT_DIR, MODEL_DIR
from metrics import compute_all_metrics

RUNS = {
    "baseline":   dict(latent_dim=256, filter_size=3, num_layers=3, activation="elu",        decoder_type="interpolation", num_res_blocks=2),
    "latent_64":  dict(latent_dim=64,  filter_size=3, num_layers=3, activation="elu",        decoder_type="interpolation", num_res_blocks=2),
    "latent_512": dict(latent_dim=512, filter_size=3, num_layers=3, activation="elu",        decoder_type="interpolation", num_res_blocks=2),
    "filter_5":   dict(latent_dim=256, filter_size=5, num_layers=3, activation="elu",        decoder_type="interpolation", num_res_blocks=2),
    "filter_7":   dict(latent_dim=256, filter_size=7, num_layers=3, activation="elu",        decoder_type="interpolation", num_res_blocks=2),
    "deconv":     dict(latent_dim=256, filter_size=3, num_layers=3, activation="elu",        decoder_type="deconv",        num_res_blocks=2),
    "act_relu":   dict(latent_dim=256, filter_size=3, num_layers=3, activation="relu",       decoder_type="interpolation", num_res_blocks=2),
    "act_leaky":  dict(latent_dim=256, filter_size=3, num_layers=3, activation="leaky_relu", decoder_type="interpolation", num_res_blocks=2),
    "layers_2":   dict(latent_dim=256, filter_size=3, num_layers=2, activation="elu",        decoder_type="interpolation", num_res_blocks=2),
    "layers_4":   dict(latent_dim=256, filter_size=3, num_layers=4, activation="elu",        decoder_type="interpolation", num_res_blocks=2),
    "no_res":     dict(latent_dim=256, filter_size=3, num_layers=3, activation="elu",        decoder_type="interpolation", num_res_blocks=0),
    "res_1":      dict(latent_dim=256, filter_size=3, num_layers=3, activation="elu",        decoder_type="interpolation", num_res_blocks=1),
}


def get_val_loader():
    val_dataset = FacesDataset(augment=False)
    val_size    = int(len(val_dataset) * 0.1)
    train_size  = len(val_dataset) - val_size
    _, val_idx  = random_split(list(range(len(val_dataset))), [train_size, val_size],
                               generator=torch.Generator().manual_seed(42))
    val_set     = Subset(val_dataset, list(val_idx))
    return DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)


val_loader  = get_val_loader()
all_results = {}

for run_name, cfg in RUNS.items():
    ckpt = os.path.join(MODEL_DIR, "ablations", run_name, "best.pth")
    if not os.path.exists(ckpt):
        print(f"[SKIP] {run_name}")
        continue

    print(f"\nEvaluating {run_name}...")
    model = VAE(num_classes=2, **cfg).to(DEVICE)
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    model.eval()

    all_recon, all_real = [], []
    with torch.no_grad():
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

        all_recon  = torch.cat(all_recon)
        all_real   = torch.cat(all_real)
        glasses    = model.generate(label=0, n=64, device=DEVICE)
        no_glasses = model.generate(label=1, n=64, device=DEVICE)
        gen        = torch.cat([glasses, no_glasses])

    m = compute_all_metrics(all_recon, all_real, gen, DEVICE)
    all_results[run_name] = m
    print(f"  SSIM:{m['SSIM']:.4f} LPIPS:{m['LPIPS']:.4f} FID:{m['FID']:.2f} IS:{m['IS']:.4f} MMD:{m['MMD']:.6f} P:{m['Precision']:.4f} R:{m['Recall']:.4f}")

print(f"\n{'='*90}")
print(f"{'Run':<20} {'SSIM':>6} {'LPIPS':>7} {'FID':>8} {'IS':>7} {'MMD':>10} {'Prec':>7} {'Rec':>7}")
print("-" * 90)
for run_name, m in all_results.items():
    print(f"{run_name:<20} {m['SSIM']:>6.4f} {m['LPIPS']:>7.4f} {m['FID']:>8.2f} {m['IS']:>7.4f} {m['MMD']:>10.6f} {m['Precision']:>7.4f} {m['Recall']:>7.4f}")

os.makedirs(os.path.join(OUTPUT_DIR, "ablations"), exist_ok=True)
csv_path = os.path.join(OUTPUT_DIR, "ablations", "full_metrics.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["run", "SSIM", "LPIPS", "FID", "IS", "MMD", "Precision", "Recall"])
    writer.writeheader()
    for run_name, m in all_results.items():
        writer.writerow({"run": run_name, **{k: f"{v:.4f}" for k, v in m.items()}})

print(f"\nSaved to {csv_path}")