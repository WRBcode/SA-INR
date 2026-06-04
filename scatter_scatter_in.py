import torch
import numpy as np
from numpy.ma.core import shape


def HG_phase(cos_theta, g=0.9):
    numerator = 1 - g**2
    denominator = (1 + g**2 + 2*g*cos_theta)**1.5
    return (1/(4*np.pi)) * numerator / denominator

def compute_scatter_term(intensity_pre, I0, voxel_size,
                                   is_metal_region, metal_albedo=0.5, tissue_albedo=0.9, g=0.9):
    batch, num_rays, num_samples, num_energy = intensity_pre.shape

    albedo = torch.where(is_metal_region.unsqueeze(-1), metal_albedo, tissue_albedo)

    sigma_s = albedo * intensity_pre  # 散射截面

    # 累积和优化：计算从源到每个点的衰减
    cumsum_intensity = torch.cumsum(intensity_pre, dim=2)  # shape: (batch, num_rays, num_samples, num_energy)
    L_x = I0.view(1, 1, 1, num_energy) * torch.exp(-voxel_size * cumsum_intensity)

    # HG相函数（常量）前向散射，cos_theta≈1
    p_hg = HG_phase(1.0, g=g)

    # 散射贡献：一次性计算所有 idx 而不是 for 循环
    scatter_contrib_all = sigma_s * L_x * p_hg * voxel_size  # shape: (batch, num_rays, num_samples, num_energy)

    # 在 sample 维度上累加所有散射贡献
    scatter_term = torch.sum(scatter_contrib_all, dim=2)  # shape: (batch, num_rays, num_energy)

    return scatter_term

