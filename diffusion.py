import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, timesteps)


def cosine_beta_schedule(timesteps, s=0.008):
    steps = torch.arange(timesteps + 1, dtype=torch.float64)
    alphas_cumprod = torch.cos(((steps / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 0.0, 0.999).float()


class SinusoidalPosEmb(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        return F.normalize(x, dim=1) * self.g * (x.shape[1] ** 0.5)


class Block(nn.Module):
    def __init__(self, dim_in, dim_out, use_context=False):
        super().__init__()
        self.conv = nn.Conv2d(dim_in, dim_out, 3, padding=1)
        self.norm = RMSNorm(dim_out)
        self.act = nn.GELU()
        self.use_context = use_context

    def forward(self, x, scale_shift=None):
        x = self.conv(x)
        x = self.norm(x)
        if scale_shift is not None:
            scale, shift = scale_shift
            x = x * (scale + 1) + shift
        return self.act(x)


class ResnetBlock(nn.Module):

    def __init__(self, dim_in, dim_out, context_dim, dropout=0.1):
        super().__init__()
        self.block1 = Block(dim_in, dim_out)
        self.block2 = Block(dim_out, dim_out)
        self.res_conv = nn.Conv2d(dim_in, dim_out, 1) if dim_in != dim_out else nn.Identity()
        self.context_mlp = nn.Sequential(
            nn.GELU(),
            nn.Linear(context_dim, dim_out * 2),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context):
        h = self.block1(x)
        ctx = self.context_mlp(context)[:, :, None, None]
        scale, shift = ctx.chunk(2, dim=1)
        h = self.block2(h, scale_shift=(scale, shift))
        h = self.dropout(h)
        return h + self.res_conv(x)


class Downsample(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.conv = nn.Conv2d(dim_in, dim_out, kernel_size=2, stride=2)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = nn.Conv2d(dim_in, dim_out, 3, padding=1)

    def forward(self, x):
        return self.conv(self.up(x))

class UNet(nn.Module):

    def __init__(self, dim=64, num_classes=2, dim_mults=(1, 2, 4, 8),
                 channels=3, dropout=0.1):
        super().__init__()
        self.channels = channels
        context_dim = dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, context_dim),
            nn.GELU(),
            nn.Linear(context_dim, context_dim),
        )
        self.class_emb = nn.Embedding(num_classes + 1, context_dim)

        self.init_conv = nn.Conv2d(channels, dim, 7, padding=3)

        self.downs = nn.ModuleList()
        dims = [dim]
        ch = dim
        for mult in dim_mults:
            out_ch = dim * mult
            self.downs.append(nn.ModuleList([
                ResnetBlock(ch, out_ch, context_dim, dropout),
                ResnetBlock(out_ch, out_ch, context_dim, dropout),
                Downsample(out_ch, out_ch),
            ]))
            dims.append(out_ch)
            ch = out_ch

        self.mid1 = ResnetBlock(ch, ch, context_dim, dropout)
        self.mid2 = ResnetBlock(ch, ch, context_dim, dropout)

        self.ups = nn.ModuleList()
        for mult in reversed(dim_mults):
            out_ch = dim * mult
            self.ups.append(nn.ModuleList([
                Upsample(ch, out_ch),
                ResnetBlock(out_ch + dims.pop(), out_ch, context_dim, dropout),
                ResnetBlock(out_ch, out_ch, context_dim, dropout),
            ]))
            ch = out_ch

        self.final = nn.Sequential(
            RMSNorm(ch),
            nn.GELU(),
            nn.Conv2d(ch, channels, 1),
        )

    def forward(self, x, t, class_label):
        cls_idx = class_label.clone().long()
        cls_idx[cls_idx < 0] = self.class_emb.num_embeddings - 1
        ctx = self.time_mlp(t) + self.class_emb(cls_idx)

        x = self.init_conv(x)
        skips = [x]

        for resblock1, resblock2, down in self.downs:
            x = resblock1(x, ctx)
            x = resblock2(x, ctx)
            skips.append(x)
            x = down(x)

        x = self.mid1(x, ctx)
        x = self.mid2(x, ctx)

        for up, resblock1, resblock2 in self.ups:
            x = up(x)
            x = torch.cat([x, skips.pop()], dim=1)
            x = resblock1(x, ctx)
            x = resblock2(x, ctx)

        return self.final(x)


class GaussianDiffusion(nn.Module):

    def __init__(self, model, timesteps=1000, beta_schedule="linear",
                 beta_start=1e-4, beta_end=0.02):
        super().__init__()
        self.model = model
        self.timesteps = timesteps

        if beta_schedule == "linear":
            betas = linear_beta_schedule(timesteps, beta_start, beta_end)
        elif beta_schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f"Unknown schedule: {beta_schedule}")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))

        posterior_var = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_var", posterior_var)
        self.register_buffer("posterior_mean_coef1",
                             betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        self.register_buffer("posterior_mean_coef2",
                             (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod))

    def _extract(self, buf, t, x_shape):
        out = buf.gather(-1, t)
        return out.reshape(-1, *((1,) * (len(x_shape) - 1)))

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alpha = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        return sqrt_alpha * x_start + sqrt_one_minus * noise

    def p_losses(self, x_start, t, class_label, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start, t, noise)
        predicted_noise = self.model(x_noisy, t, class_label)
        return F.mse_loss(predicted_noise, noise)

    def forward(self, x, class_label):
        b = x.shape[0]
        t = torch.randint(0, self.timesteps, (b,), device=x.device).long()
        return self.p_losses(x, t, class_label)

    def _predict_x0_from_noise(self, x_t, t, noise):
        sqrt_alpha = self._extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_one_minus = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        return (x_t - sqrt_one_minus * noise) / sqrt_alpha

    @torch.no_grad()
    def p_sample(self, x, t_index, class_label):
        b = x.shape[0]
        t = torch.full((b,), t_index, device=x.device, dtype=torch.long)

        predicted_noise = self.model(x, t, class_label)

        x0_pred = self._predict_x0_from_noise(x, t, predicted_noise)
        x0_pred = x0_pred.clamp(-1, 1)

        coef1 = self._extract(self.posterior_mean_coef1, t, x.shape)
        coef2 = self._extract(self.posterior_mean_coef2, t, x.shape)
        model_mean = coef1 * x0_pred + coef2 * x

        if t_index == 0:
            return model_mean
        else:
            posterior_var = self._extract(self.posterior_var, t, x.shape)
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_var) * noise

    @torch.no_grad()
    def sample(self, class_label, image_size, channels=3):

        b = class_label.shape[0]
        device = next(self.model.parameters()).device
        class_label = class_label.to(device)

        x = torch.randn(b, channels, image_size, image_size, device=device)

        for t_index in reversed(range(self.timesteps)):
            x = self.p_sample(x, t_index, class_label)

        return (x.clamp(-1, 1) + 1) / 2
