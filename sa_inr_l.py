"""
SA-INR-L (learnable variant), robust re-implementation.

= the working SA-INR-P (Polyner_sc.py) forward model, but the two albedo
values become learnable per-scan scalars (alpha_metal, alpha_tissue),
initialized at the NIST-motivated (0.5, 0.9). A tiny additive MLP with a
ZERO-initialized final layer provides an optional per-sample correction
that starts as an exact no-op, so at initialization SA-INR-L is identical
to SA-INR-P and can only deviate if it reduces the loss. This keeps the
extra capacity minimal (avoiding the over-fitting seen in the
capacity-matched control) while making the scattering parameters adaptive.
"""
import model_sc
import torch
import torch.nn as nn
import numpy as np
import dataset
import SimpleITK as sitk
import tinycudann as tcnn
from tqdm import tqdm
from torch.utils import data
from scipy import io as scio
from torch.optim import lr_scheduler
from utils import prepare_mask_for_scatter


def HG_phase(cos_theta, g=0.9):
    num = 1 - g ** 2
    den = (1 + g ** 2 + 2 * g * cos_theta) ** 1.5
    return (1 / (4 * np.pi)) * num / den


class ScatterMLP(nn.Module):
    """2 -> 64 -> 64 -> 1, ReLU; final layer zero-init => starts as a no-op."""
    def __init__(self, hidden=64):
        super().__init__()
        self.l1 = nn.Linear(2, hidden)
        self.l2 = nn.Linear(hidden, hidden)
        self.l3 = nn.Linear(hidden, 1)
        nn.init.zeros_(self.l3.weight)
        nn.init.zeros_(self.l3.bias)

    def forward(self, x):
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        return self.l3(x)


def compute_scatter_learn(intensity_pre, I0, voxel_size, is_metal_region,
                          metal_albedo, tissue_albedo, delta, g=0.9):
    b, r, s, e = intensity_pre.shape
    base = torch.where(is_metal_region.unsqueeze(-1), metal_albedo, tissue_albedo)
    albedo = torch.clamp(base + delta, min=0.0)          # delta ~ 0 at init
    sigma_s = albedo * intensity_pre
    cumsum = torch.cumsum(intensity_pre, dim=2)
    L_x = I0.view(1, 1, 1, e) * torch.exp(-voxel_size * cumsum)
    contrib = sigma_s * L_x * HG_phase(1.0, g=g) * voxel_size
    return torch.sum(contrib, dim=2)


def train(img_id, config):
    in_path = config["file"]["in_dir"]
    out_path = config["file"]["out_dir"]
    model_path = config["file"]["model_dir"]
    proj_path = "{}/ma_sinogram_{}.nii".format(in_path, img_id)
    proj_pos_path = "{}/fanSensorPos.nii".format(in_path)
    mask_path = "{}/mask_{}.nii".format(in_path, img_id)
    h, w, SOD = config["file"]["h"], config["file"]["w"], config["file"]["SOD"]
    voxel_size = config["file"]["voxel_size"]
    num_angle, _ = sitk.GetArrayFromImage(sitk.ReadImage(proj_path)).shape
    lr = config["train"]["lr"]
    gpu = config["train"]["gpu"]
    epoch = config["train"]["epoch"]
    save_epoch = config["train"]["save_epoch"]
    lr_decay_epoch = config["train"]["lr_decay_epoch"]
    lr_decay_coefficient = config["train"]["lr_decay_coefficient"]
    batch_size = config["train"]["batch_size"]
    num_sample_ray = config["train"]["num_sample_ray"]
    lamb = config["train"]["lambda"]
    g_value = config["train"].get("g", 0.9)
    device = torch.device("cuda:{}".format(str(gpu) if torch.cuda.is_available() else "cpu"))

    mask = sitk.GetArrayFromImage(sitk.ReadImage(mask_path))
    mask = np.rot90(np.pad(mask, ((int(SOD - mask.shape[0] / 2), int(SOD - mask.shape[0] / 2) - 1),
                                  (int(SOD - mask.shape[1] / 2), int(SOD - mask.shape[1] / 2) - 1)))).copy()
    mask_tensor = torch.tensor(mask).float().unsqueeze(0).unsqueeze(0).to(device)
    mask_tensor = torch.where(mask_tensor == 1, 0.0, 1.0)

    spectrum_ini = scio.loadmat("./{}/GE14Spectrum120KVP.mat".format(in_path))["GE14Spectrum120KVP"]
    e_1, e_n = (20, 120)
    spectrum_ini_np = spectrum_ini[(e_1 - 1):e_n, 1]
    e_level = len(spectrum_ini_np)
    spectrum_ini_tensor = torch.tensor(spectrum_ini_np, dtype=torch.float).to(device)
    spectrum = (spectrum_ini_tensor / torch.sum(spectrum_ini_tensor)).view(1, 1, -1)

    dc_loss = torch.nn.L1Loss().to(device)
    ase_loss = model_sc.Attenuation_Smootion_Over_Energies_Loss(lamb=lamb, mask=mask_tensor).to(device)

    network = tcnn.NetworkWithInputEncoding(
        n_input_dims=2, n_output_dims=e_level,
        encoding_config=config["encoding"], network_config=config["network"]).to(device)
    scatter_mlp = ScatterMLP(64).to(device)
    alpha_metal = torch.nn.Parameter(torch.tensor(0.5, device=device))
    alpha_tissue = torch.nn.Parameter(torch.tensor(0.9, device=device))

    optimizer = torch.optim.Adam(
        list(network.parameters()) + list(scatter_mlp.parameters()) + [alpha_metal, alpha_tissue], lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_decay_epoch, gamma=lr_decay_coefficient)

    train_loader = data.DataLoader(
        dataset=dataset.TrainData(proj_path=proj_path, proj_pos_path=proj_pos_path, SOD=SOD,
                                  num_sample_ray=num_sample_ray, num_angle=num_angle, voxel_size=voxel_size),
        batch_size=batch_size, shuffle=True)
    test_loader = data.DataLoader(dataset=dataset.TestData(h=(2 * SOD + 1), w=(2 * SOD + 1)),
                                  batch_size=1, shuffle=False)

    loop_tqdm = tqdm(range(epoch), leave=False)
    for e in loop_tqdm:
        network.train(); scatter_mlp.train()
        loss_log = 0
        for i, (ray, proj) in enumerate(train_loader):
            ray = ray.to(device).float().view(-1, 2)
            proj = proj.to(device).float()
            intensity_pre = network(ray).view(-1, num_sample_ray, 2 * SOD, e_level).float()
            proj_pre = torch.exp(-voxel_size * torch.sum(intensity_pre, dim=2).squeeze(-1))
            is_metal_region = prepare_mask_for_scatter(mask_tensor, intensity_pre)
            delta = scatter_mlp(ray).view(-1, num_sample_ray, 2 * SOD, 1)
            scatter_pre = compute_scatter_learn(
                intensity_pre, spectrum_ini_tensor, voxel_size, is_metal_region,
                alpha_metal, alpha_tissue, delta, g=g_value)
            actual_det = proj_pre - scatter_pre
            actual_det = -torch.log(torch.sum((actual_det * spectrum), dim=-1).squeeze(-1))
            loss = dc_loss(actual_det, proj.to(actual_det.dtype)) + ase_loss(intensity=intensity_pre, ray=ray)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            loss_log += loss.item()
        else:
            scheduler.step()
            loop_tqdm.set_description("Image #{}".format(img_id))
            loop_tqdm.set_postfix(lr=scheduler.get_last_lr()[0], loss=loss_log / len(train_loader))

        if (e + 1) % save_epoch == 0:
            kx, ky = int(1 + (2 * SOD - h) / 2), int((2 * SOD - w) / 2)
            with torch.no_grad():
                torch.save(network.state_dict(), f"{model_path}/model_{img_id}.pkl")
                for i, xy in enumerate(test_loader):
                    xy = xy.to(device).float().view(-1, 2)
                    output = network(xy)
                    img_pre = output[:, int(e_level // 2)].view(2 * SOD + 1, 2 * SOD + 1)
                    img_pre_np = img_pre.cpu().detach().numpy().astype(np.float32)[kx:kx + h, ky:ky + w]
                    if img_pre_np.shape[1] > 0:
                        img_pre_np = np.flip(img_pre_np, axis=1)
                sitk.WriteImage(sitk.GetImageFromArray(img_pre_np), f"{out_path}/polyner_{img_id}.nii")
