import argparse
import os

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import matplotlib.pyplot as plt
import numpy as np

import datasets
from positional_embeddings import PositionalEmbedding


class Block(nn.Module):
    def __init__(self, size: int):
        super().__init__()

        self.ff = nn.Linear(size, size)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor):
        return x + self.act(self.ff(x))


class MLP(nn.Module):
    def __init__(self, hidden_size: int = 128, hidden_layers: int = 3, emb_size: int = 128,
                 time_emb: str = "sinusoidal", input_emb: str = "sinusoidal", input_dim: int = 2):
        super().__init__()

        self.time_mlp = PositionalEmbedding(emb_size, time_emb)
        self.input_mlp1 = PositionalEmbedding(emb_size, input_emb, scale=25.0)
        self.input_dim = input_dim
        if self.input_dim == 2:
            self.input_mlp2 = PositionalEmbedding(emb_size, input_emb, scale=25.0)

            concat_size = len(self.time_mlp.layer) + \
                len(self.input_mlp1.layer) + len(self.input_mlp2.layer)
        else:
            concat_size = len(self.time_mlp.layer) + len(self.input_mlp1.layer)
        layers = [nn.Linear(concat_size, hidden_size).double(), nn.GELU().double()]
        for _ in range(hidden_layers):
            layers.append(Block(hidden_size).double())
        layers.append(nn.Linear(hidden_size, self.input_dim).double())
        self.joint_mlp = nn.Sequential(*layers).double()

    def forward(self, x, t):
        if self.input_dim == 2:
            x1_emb = self.input_mlp1(x[:, 0])
            x2_emb = self.input_mlp2(x[:, 1])
            t_emb = self.time_mlp(t)
            x = torch.cat((x1_emb, x2_emb, t_emb), dim=-1)
            x = self.joint_mlp(x)
        else:
            x1_emb = self.input_mlp1(x[:, 0])
            t_emb = self.time_mlp(t)
            x = torch.cat((x1_emb, t_emb), dim=-1)
            x = self.joint_mlp(x)
        return x

def find_scaling(t_big, t_small, num_steps):
    s_min = 0.001
    s_max = 1
    while True:
        s_curr = np.sqrt(s_min * s_max)
        curr_time = t_big
        times = []
        step_size = 0
        while curr_time > t_small:
            step_size = s_curr * (1. - np.exp(-2. * curr_time))
            times.append(curr_time)
            curr_time -= step_size
        curr_time += step_size
        if len(times) < num_steps:
            s_max = s_curr
        elif len(times) > num_steps:
            s_min = s_curr
        elif abs(curr_time - t_small) > 1e-6:
            s_min = s_curr
        else:
            break
    return np.sqrt(s_min * s_max)

def calculate_betas(num_timesteps, num_our_steps, beta_start, beta_end):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    original_betas = torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float64)
    alphas = 1.0 - original_betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    original_times = -0.5 * torch.log(alphas_cumprod)
    big_time = original_times[-1].item()
    small_time = original_times[0].item()
    step_size_eps = find_scaling(big_time, small_time, num_our_steps)
    curr_time = original_times[-1]
    betas = []
    times = []
    step_sizes = []
    step_size_eps = np.float64(step_size_eps)
    curr_time = np.float64(curr_time)
    while curr_time > 1e-9:
        step_size = step_size_eps * (1. - np.exp(-2. * curr_time))
        step_sizes.append(step_size.item())
        times.append(curr_time.item())
        betas.append(1 - np.exp(-2. * step_size))
        # print(curr_time, step_size)
        curr_time -= step_size
    alphas = 1.0 - torch.tensor(betas[::-1], dtype=torch.float64)
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    original_times = -0.5 * torch.log(alphas_cumprod)
    print("MIN GAMMA", np.sqrt(1 - np.exp(-2*original_times[0])), "LEN", len(original_times))
    return torch.tensor(betas[::-1], dtype=torch.float64, device=device), original_times.to(device)

class NoiseScheduler():
    def __init__(self,
                 num_timesteps=1000,
                 beta_start=0.0001,
                 beta_end=0.02,
                 beta_schedule="linear"):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.num_timesteps = num_timesteps
        if beta_schedule == "linear":
            self.betas = torch.linspace(
                beta_start, beta_end, num_timesteps, dtype=torch.float64, device=device)
        elif beta_schedule == "quadratic":
            self.betas = torch.linspace(
                beta_start ** 0.5, beta_end ** 0.5, num_timesteps, dtype=torch.float64, device=device) ** 2
        elif beta_schedule == "ours":
            self.betas, self.times = calculate_betas(num_timesteps, 50, beta_start, beta_end)
        self.num_timesteps = len(self.betas)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0).to(device)
        self.alphas_cumprod_prev = F.pad(
            self.alphas_cumprod[:-1], (1, 0), value=1.)

        # required for self.add_noise
        self.sqrt_alphas_cumprod = self.alphas_cumprod ** 0.5
        self.sqrt_one_minus_alphas_cumprod = (1 - self.alphas_cumprod) ** 0.5

        # required for reconstruct_x0
        self.sqrt_inv_alphas_cumprod = torch.sqrt(1 / self.alphas_cumprod)
        self.sqrt_inv_alphas_cumprod_minus_one = torch.sqrt(
            1 / self.alphas_cumprod - 1)

        # required for q_posterior
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1. - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1. - self.alphas_cumprod)

    def reconstruct_x0(self, x_t, t, noise):
        s1 = self.sqrt_inv_alphas_cumprod[t]
        s2 = self.sqrt_inv_alphas_cumprod_minus_one[t]
        s1 = s1.reshape(-1, 1)
        s2 = s2.reshape(-1, 1)
        return s1 * x_t - s2 * noise

    def q_posterior(self, x_0, x_t, t):
        s1 = self.posterior_mean_coef1[t]
        s2 = self.posterior_mean_coef2[t]
        s1 = s1.reshape(-1, 1)
        s2 = s2.reshape(-1, 1)
        # print("S1, S2", s1, s2)
        mu = s1 * x_0 + s2 * x_t
        return mu

    def get_variance(self, t):
        if t == 0:
            return 0

        variance = self.betas[t] * (1. - self.alphas_cumprod_prev[t]) / (1. - self.alphas_cumprod[t])
        variance = variance.clip(1e-20)
        return variance

    def step(self, model_output, timestep, sample, noise=True):
        t = timestep
        pred_original_sample = self.reconstruct_x0(sample, t, model_output)
        pred_prev_sample = self.q_posterior(pred_original_sample, sample, t)
        # print(sample, pred_original_sample, pred_prev_sample)
        variance = 0
        if t > 0:
            noise = torch.randn_like(model_output)
            variance = (self.get_variance(t) ** 0.5) * noise
        else:
            noise = torch.randn_like(model_output)
            variance = (self.get_variance(t) ** 0.5) * noise
            # print(noise, self.get_variance(t)**0.5)
            # print("LOL")
        pred_prev_sample = pred_prev_sample + variance
    
        return pred_prev_sample
    def add_noise(self, x_start, x_noise, timesteps):
        s1 = self.sqrt_alphas_cumprod[timesteps]
        s2 = self.sqrt_one_minus_alphas_cumprod[timesteps]
        s1 = s1.reshape(-1, 1)
        s2 = s2.reshape(-1, 1)
        return s1 * x_start + s2 * x_noise

    def __len__(self):
        return self.num_timesteps

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, default="base")
    parser.add_argument("--dataset", type=str, default="dino", choices=["circle", "dino", "line", "moons", "square", "point1d"])
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=1000)
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--num_timesteps", type=int, default=50)
    parser.add_argument("--beta_schedule", type=str, default="linear", choices=["linear", "quadratic", "ours"])
    parser.add_argument("--embedding_size", type=int, default=128)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--hidden_layers", type=int, default=3)
    parser.add_argument("--time_embedding", type=str, default="sinusoidal", choices=["sinusoidal", "learnable", "linear", "zero"])
    parser.add_argument("--input_embedding", type=str, default="sinusoidal", choices=["sinusoidal", "learnable", "linear", "identity"])
    parser.add_argument("--save_images_step", type=int, default=1)
    parser.add_argument("--dataset_size", type=int, default=8000)
    parser.add_argument("--dimension", type=int, default=2)
    config = parser.parse_args()

    dataset = datasets.get_dataset(config.dataset, n=config.dataset_size)
    dataloader = DataLoader(
        dataset, batch_size=config.train_batch_size, shuffle=True, drop_last=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(
        hidden_size=config.hidden_size,
        hidden_layers=config.hidden_layers,
        emb_size=config.embedding_size,
        time_emb=config.time_embedding,
        input_emb=config.input_embedding,
        input_dim=config.dimension).to(device)

    noise_scheduler = NoiseScheduler(
        num_timesteps=config.num_timesteps,
        beta_schedule=config.beta_schedule)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
    )

    global_step = 0
    frames = []
    losses = []
    print("Training model...")
    for epoch in range(config.num_epochs):
        model.train()
        progress_bar = tqdm(total=len(dataloader))
        progress_bar.set_description(f"Epoch {epoch}")
        for step, batch in enumerate(dataloader):
            batch = batch[0].to(device)
            noise = torch.randn(batch.shape, dtype=torch.float64).to(device)
            timesteps = torch.randint(
                0, noise_scheduler.num_timesteps, (batch.shape[0],)
            ).long().to(device)
            noisy = noise_scheduler.add_noise(batch, noise, timesteps)
            noise_pred = model(noisy, timesteps)
            # stds = torch.sqrt(torch.sqrt(1 - torch.exp(-2 * noise_scheduler.times[timesteps]))).detach()
            loss = F.mse_loss(noise_pred, noise)
            # loss = F.mse_loss((noise_pred - noise).squeeze() / stds, torch.zeros_like((noise_pred - noise).squeeze() / stds))
            # print(loss.dtype, stds.dtype)
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "step": global_step}
            losses.append(loss.detach().item())
            progress_bar.set_postfix(**logs)
            global_step += 1
        progress_bar.close()

        if epoch % config.save_images_step == 0 or epoch == config.num_epochs - 1:
            # generate data with the model to later visualize the learning process
            model.eval()
            sample = torch.randn(config.eval_batch_size, 2).to(device)
            timesteps = list(range(len(noise_scheduler)))[::-1]
            for i, t in enumerate(tqdm(timesteps)):
                t = torch.from_numpy(np.repeat(t, config.eval_batch_size)).long().to(device)
                with torch.no_grad():
                    residual = model(sample, t)
                sample = noise_scheduler.step(residual, t[0], sample)
            sample_cpu = sample.cpu()
            frames.append(sample_cpu.numpy())

    print("Saving model...")
    outdir = f"exps/{config.experiment_name}"
    os.makedirs(outdir, exist_ok=True)
    torch.save(model.state_dict(), f"{outdir}/model.pth")

    print("Saving images...")
    imgdir = f"{outdir}/images"
    os.makedirs(imgdir, exist_ok=True)
    frames = np.stack(frames)
    xmin, xmax = -6, 6
    ymin, ymax = -6, 6
    for i, frame in enumerate(frames):
        plt.figure(figsize=(10, 10))
        plt.scatter(frame[:, 0], frame[:, 1])
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        plt.savefig(f"{imgdir}/{i:04}.png")
        plt.close()

    print("Saving loss as numpy array...")
    np.save(f"{outdir}/loss.npy", np.array(losses))

    print("Saving frames...")
    np.save(f"{outdir}/frames.npy", frames)

# class NewNoiseScheduler():
#     def __init__(self,
#                  num_timesteps=1000,
#                  beta_start=0.0001,
#                  beta_end=0.02,
#                  beta_schedule="linear"):
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#         self.betas, self.times = calculate_betas(num_timesteps, 50, beta_start, beta_end)
#         self.num_timesteps = len(self.betas)
#         self.alphas = 1.0 - self.betas
#         self.alphas_cumprod = torch.cumprod(self.alphas, axis=0).to(device)
#         self.alphas_cumprod_prev = F.pad(
#             self.alphas_cumprod[:-1], (1, 0), value=1.)

#         # required for self.add_noise
#         self.sqrt_alphas_cumprod = self.alphas_cumprod ** 0.5
#         self.sqrt_one_minus_alphas_cumprod = (1 - self.alphas_cumprod) ** 0.5

#         # required for reconstruct_x0
#         self.sqrt_inv_alphas_cumprod = torch.sqrt(1 / self.alphas_cumprod)
#         self.sqrt_inv_alphas_cumprod_minus_one = torch.sqrt(
#             1 / self.alphas_cumprod - 1)

#         # required for q_posterior
#         self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
#         self.posterior_mean_coef2 = (1. - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1. - self.alphas_cumprod)

#         self.variances = 1 - torch.exp(-2 * self.times)

#     def step(self, model_output, timestep, sample, noise=True, noise_step=True):
#         et = 1 / self.alphas[timestep]
#         if noise_step:
#             no_noise = et * sample + 2 * model_output * (et - 1) *  torch.sqrt(self.variances[timestep])
#             noise_level = torch.randn_like(model_output) * torch.sqrt(self.variances[timestep]) * et
#             return no_noise + noise_level
#         else:
#             no_noise = et * sample + model_output * (et - 1) *  torch.sqrt(self.variances[timestep])
#             return no_noise

#     def add_noise(self, x_start, x_noise, timesteps):
#         s1 = self.sqrt_alphas_cumprod[timesteps]
#         s2 = self.sqrt_one_minus_alphas_cumprod[timesteps]
#         s1 = s1.reshape(-1, 1)
#         s2 = s2.reshape(-1, 1)
#         return s1 * x_start + s2 * x_noise

#     def __len__(self):
#         return self.num_timesteps