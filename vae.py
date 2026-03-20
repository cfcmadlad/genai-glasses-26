import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):

    def __init__(self, latent_dim=128, num_classes=2, filter_size=3,
                 num_layers=3, activation="relu", decoder_type="deconv"):
        super().__init__()

        self.latent_dim  = latent_dim
        self.num_classes = num_classes

        acts = {
            "relu":       nn.ReLU(),
            "leaky_relu": nn.LeakyReLU(0.2),
            "elu":        nn.ELU()
        }
        act = acts[activation]
        pad = filter_size // 2


        enc_channels = [3] + [32 * (2 ** i) for i in range(num_layers)]

        enc_layers = []
        for i in range(num_layers):
            enc_layers.append(nn.Conv2d(enc_channels[i], enc_channels[i+1],
                                        filter_size, stride=2, padding=pad))
            enc_layers.append(nn.BatchNorm2d(enc_channels[i+1]))
            enc_layers.append(act)

        self.encoder        = nn.Sequential(*enc_layers)
        self.final_channels = enc_channels[-1]
        self.spatial        = 64 // (2 ** num_layers)
        flat_dim            = self.final_channels * self.spatial * self.spatial

        self.fc_mu     = nn.Linear(flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(flat_dim, latent_dim)
        self.fc_decode = nn.Linear(latent_dim + num_classes, flat_dim)


        dec_channels = [self.final_channels // (2 ** i) for i in range(num_layers)] + [3]

        dec_layers = []
        for i in range(num_layers):
            is_last = (i == num_layers - 1)

            if decoder_type == "deconv":
                dec_layers.append(
                    nn.ConvTranspose2d(dec_channels[i], dec_channels[i+1],
                                      filter_size, stride=2, padding=pad, output_padding=1)
                )
            else:
                dec_layers.append(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False))
                dec_layers.append(nn.Conv2d(dec_channels[i], dec_channels[i+1],
                                            filter_size, padding=pad))

            if not is_last:
                dec_layers.append(nn.BatchNorm2d(dec_channels[i+1]))
                dec_layers.append(act)

        dec_layers.append(nn.Tanh())
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x):
        h      = self.encoder(x).flatten(1)
        mu     = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, labels):
        label_oh = F.one_hot(labels, self.num_classes).float()
        x        = torch.cat([z, label_oh], dim=1)
        x        = self.fc_decode(x)
        x        = x.view(x.size(0), self.final_channels, self.spatial, self.spatial)
        return self.decoder(x)

    def forward(self, x, labels):
        mu, logvar = self.encode(x)
        z          = self.reparameterize(mu, logvar)
        recon      = self.decode(z, labels)
        return recon, mu, logvar

    def generate(self, label, n, device):
        self.eval()
        with torch.no_grad():
            z      = torch.randn(n, self.latent_dim).to(device)
            labels = torch.full((n,), label, dtype=torch.long).to(device)
            imgs   = self.decode(z, labels)
        return imgs


def vae_loss(recon, x, mu, logvar, beta=1.0):
    recon_loss = F.mse_loss(recon, x, reduction="mean")
    kl_loss    = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    total      = recon_loss + beta * kl_loss
    return total, recon_loss, kl_loss