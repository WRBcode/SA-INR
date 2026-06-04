import numpy as np, SimpleITK as sitk
from tqdm import tqdm
from skimage.metrics import structural_similarity
from skimage.metrics import peak_signal_noise_ratio

def psnr(image, ground_truth):
    data_range = np.max(ground_truth) - np.min(ground_truth)
    return peak_signal_noise_ratio(ground_truth, image, data_range=data_range)


def ssim(image, ground_truth):
    data_range = np.max(ground_truth) - np.min(ground_truth)
    return structural_similarity(image, ground_truth, data_range=data_range)


def fan_beam_ray(proj_pos, SOD):
    origin_x = 0
    origin_y = -1
    y = np.linspace(-1, 1, int(2 * SOD)).reshape(-1, 1)
    x = np.zeros_like(y)
    xy_temp = np.concatenate((x, y), axis=1)
    xy_temp = np.concatenate((xy_temp, np.ones_like(x)), axis=(-1))
    num_det = len(proj_pos)
    xy = np.zeros(shape=(num_det, int(2 * SOD), 2))
    for i in range(num_det):
        fan_angle_rad = np.deg2rad(proj_pos[num_det - i - 1])
        M = np.array([
         [
          np.cos(fan_angle_rad), -np.sin(fan_angle_rad),
          -1 * origin_x * np.cos(fan_angle_rad) + origin_y * np.sin(fan_angle_rad) + origin_x],
         [
          np.sin(fan_angle_rad), np.cos(fan_angle_rad),
          -1 * origin_x * np.sin(fan_angle_rad) - origin_y * np.cos(fan_angle_rad) + origin_y],
         [
          0, 0, 1]])
        temp = xy_temp @ M.T
        xy[i, :, :] = temp[:, :2]  # xy[i] shape = (2*SOD, 2)
    else:
        return xy


def grid_coordinate(h, w):
    x = np.linspace(-1, 1, h)
    y = np.linspace(-1, 1, w)
    x, y = np.meshgrid(x, y, indexing="ij")
    xy = np.stack([x, y], -1).reshape(-1, 2)
    return xy


def rotate_ray(xy, angle):
    xy_shape = xy.shape
    angle_rad = np.deg2rad(angle)
    trans_mat = np.array([
     [
      np.cos(angle_rad), -np.sin(angle_rad)],
     [
      np.sin(angle_rad), np.cos(angle_rad)]])
    xy = xy.reshape(-1, 2)
    xy = np.dot(xy, trans_mat.T).reshape(xy_shape)
    return xy

import torch

def prepare_mask_for_scatter(mask_tensor, intensity_pre):
    """
    输入:
        mask_tensor: torch.Tensor [1, 1, H, W], float类型 (0.0非金属, 1.0金属)
        intensity_pre: torch.Tensor [B, N, S, E]

    输出:
        is_metal_region: torch.BoolTensor [B, N, S]
    """
    B, N, S, _ = intensity_pre.shape

    # Step 1: pad mask 到 [1, 1, S, S]
    mask_padded = torch.nn.functional.pad(mask_tensor, (0, S - mask_tensor.shape[-1], 0, S - mask_tensor.shape[-2]))

    # Step 2: 生成金属区域 bool mask
    metal_mask = (mask_padded > 0.5).bool()  # [1, 1, S, S]
    # metal_mask = torch.where(mask_padded > 0.5, 0.99, 0.11)
    # Step 3: expand 到 batch & ray 维度 → [B, N, S]
    metal_mask_expanded = metal_mask.expand(B, N, S, S)

    # Step 4: 沿轴取中轴线的mask值作为简化（假设每ray只有一维mask）
    is_metal_region = metal_mask_expanded[:, :, S // 2, :]  # → [B, N, S]

    return is_metal_region

