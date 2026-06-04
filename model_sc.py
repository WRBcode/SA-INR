import torch
import torch.nn as nn
import torch.nn.functional as F

class Attenuation_Smootion_Over_Energies_Loss(nn.Module):
    def __init__(self, mask, lamb):
        super(Attenuation_Smootion_Over_Energies_Loss, self).__init__()
        self.mask = mask  # 需要为 shape: [1,1,H,W]
        self.lamb = lamb

    def forward(self, ray, intensity):
        batch_size, num_sample_ray, k, e_level = intensity.shape


                # 修复 mask 维度
        if self.mask.dim() == 2:
            self.mask = self.mask.unsqueeze(0).unsqueeze(0)  # → [1,1,H,W]
        elif self.mask.dim() == 3:
            self.mask = self.mask.unsqueeze(0)
        # print(f"[DEBUG] self.mask.shape (corrected) = {self.mask.shape}")

        if self.mask.dtype != torch.float32:
            self.mask = self.mask.float()
        # print(f"[DEBUG] self.mask.dtype (corrected) = {self.mask.dtype}")

        # Sample mask via grid_sample → shape: (1,1,H,W)
        sampled_mask = F.grid_sample(
            self.mask,
            ray.unsqueeze(0).unsqueeze(0),  # shape: (1,1,B,N,k,2)
            mode="nearest",
            align_corners=False
        )

        # 提取前两个维度 (1,1,...) → reshape to (B,N,k)
        mask = sampled_mask[0, 0].view(batch_size, num_sample_ray, k)

        # 计算相邻能量层差异
        diff = torch.abs(intensity[:, :, :, 1:] - intensity[:, :, :, :-1])  # shape: (B,N,k,e-1)
        diff_sum = torch.sum(diff, dim=-1) * mask  # shape: (B,N,k)

        return self.lamb * torch.sum(diff_sum) / (batch_size * num_sample_ray * k)

