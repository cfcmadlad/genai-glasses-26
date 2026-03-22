import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim
import lpips
from prdc import compute_prdc
from scipy import linalg
from torchvision.models import inception_v3
import torch.nn.functional as F

_lpips_fn = None
_inception = None

def compute_ssim(recon_imgs, real_imgs):
    gen_np  = (recon_imgs.cpu().permute(0,2,3,1).numpy() * 0.5 + 0.5).clip(0, 1)
    real_np = (real_imgs.cpu().permute(0,2,3,1).numpy()  * 0.5 + 0.5).clip(0, 1)
    n       = min(len(gen_np), len(real_np))
    scores  = [ssim(gen_np[i], real_np[i], channel_axis=2, data_range=1.0) for i in range(n)]
    return float(np.mean(scores))

def compute_lpips(recon_imgs, real_imgs, device):
    global _lpips_fn
    if _lpips_fn is None:
        _lpips_fn = lpips.LPIPS(net='vgg').to(device)
    with torch.no_grad():
        score = _lpips_fn(recon_imgs, real_imgs).mean().item()
    return score

def get_inception(device):
    global _inception
    if _inception is None:
        _inception = inception_v3(pretrained=True, transform_input=False).to(device)
        _inception.fc = torch.nn.Identity()
        _inception.eval()
    return _inception

def get_features(imgs, device):
    model = get_inception(device)
    imgs  = F.interpolate(imgs, size=(299, 299), mode='bilinear', align_corners=False)
    imgs  = (imgs * 0.5 + 0.5).clamp(0, 1)
    with torch.no_grad():
        feats = model(imgs).cpu().numpy()
    return feats

def frechet_distance(mu1, sigma1, mu2, sigma2):
    diff    = mu1 - mu2
    covmean = linalg.sqrtm(sigma1 @ sigma2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean))

def compute_fid(gen_imgs, real_imgs, device):
    gen_feats  = get_features(gen_imgs,  device)
    real_feats = get_features(real_imgs, device)
    mu_g,  sigma_g  = gen_feats.mean(0),  np.cov(gen_feats,  rowvar=False)
    mu_r,  sigma_r  = real_feats.mean(0), np.cov(real_feats, rowvar=False)
    return frechet_distance(mu_g, sigma_g, mu_r, sigma_r)

def compute_is(gen_imgs, device, splits=1):
    from torchvision.models import inception_v3 as iv3
    inc  = iv3(pretrained=True, transform_input=False).to(device).eval()
    imgs = F.interpolate(gen_imgs, size=(299, 299), mode='bilinear', align_corners=False)
    imgs = (imgs * 0.5 + 0.5).clamp(0, 1)
    with torch.no_grad():
        preds = F.softmax(inc(imgs), dim=1).cpu().numpy()
    scores = []
    for i in range(splits):
        part  = preds[i * len(preds) // splits: (i+1) * len(preds) // splits]
        py    = part.mean(axis=0)
        kl    = part * (np.log(part + 1e-10) - np.log(py + 1e-10))
        scores.append(np.exp(kl.sum(axis=1).mean()))
    return float(np.mean(scores))

def rbf_kernel(X, Y, sigma=1.0):
    XX = (X**2).sum(1, keepdims=True)
    YY = (Y**2).sum(1, keepdims=True)
    XY = X @ Y.T
    D  = XX + YY.T - 2*XY
    return np.exp(-D / (2 * sigma**2))

def compute_mmd(gen_imgs, real_imgs, device):
    gen_feats  = get_features(gen_imgs,  device)
    real_feats = get_features(real_imgs, device)
    K_gg = rbf_kernel(gen_feats,  gen_feats).mean()
    K_rr = rbf_kernel(real_feats, real_feats).mean()
    K_gr = rbf_kernel(gen_feats,  real_feats).mean()
    return float(K_gg + K_rr - 2 * K_gr)

def compute_pr(gen_imgs, real_imgs, device, k=5):
    gen_feats  = get_features(gen_imgs,  device)
    real_feats = get_features(real_imgs, device)
    metrics    = compute_prdc(real_features=real_feats, fake_features=gen_feats, nearest_k=k)
    return metrics['precision'], metrics['recall']

def compute_all_metrics(recon, real, gen, device):
    metrics = {}
    metrics['SSIM']      = compute_ssim(recon, real)
    metrics['LPIPS']     = compute_lpips(recon, real, device)
    metrics['FID']       = compute_fid(gen, real, device)
    metrics['IS']        = compute_is(gen, device)
    metrics['MMD']       = compute_mmd(gen, real, device)
    p, r                 = compute_pr(gen, real, device)
    metrics['Precision'] = p
    metrics['Recall']    = r
    return metrics