"""
SA-INR demo driver.

Reconstructs the three bundled CT slices with the two scattering-aware INR
variants from our paper. Each slice is optimized independently from its own
metal-corrupted sinogram (per-scan, self-supervised; no training set, no
pretrained weights).

  SA-INR-P (sa_inr_p.py) : physics-constrained Henyey-Greenstein single
                           scatter, fixed albedo.
  SA-INR-L (sa_inr_l.py) : learnable per-scan albedo + zero-initialized
                           scatter MLP (identical to SA-INR-P at init).

This release is intentionally limited to the three bundled examples.

Usage:
    python run_demo.py              # both variants on all 3 slices
    python run_demo.py --model p    # SA-INR-P only
    python run_demo.py --model l    # SA-INR-L only
"""
import os
import copy
import json
import argparse
import numpy as np
import SimpleITK as sitk

import sa_inr_p
import sa_inr_l
from utils import psnr, ssim

# The only examples shipped with this repository.
DEMO_IDS = [0, 1, 2]

VARIANTS = {
    "p": ("SA-INR-P", sa_inr_p, "sa_inr_p"),
    "l": ("SA-INR-L", sa_inr_l, "sa_inr_l"),
}


def evaluate(out_dir, data_dir, img_id):
    """PSNR / SSIM on non-metal regions, following the paper."""
    gt = sitk.GetArrayFromImage(sitk.ReadImage(f"{data_dir}/gt_{img_id}.nii"))
    mask = sitk.GetArrayFromImage(sitk.ReadImage(f"{data_dir}/mask_{img_id}.nii"))
    rec = sitk.GetArrayFromImage(sitk.ReadImage(f"{out_dir}/polyner_{img_id}.nii"))
    gt = np.where(mask == 1, 0, gt)
    rec = np.where(mask == 1, 0, rec)
    return psnr(image=rec, ground_truth=gt), ssim(image=rec, ground_truth=gt)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["p", "l", "both"], default="both")
    args = parser.parse_args()

    with open("config.json") as f:
        cfg = json.load(f)
    data_dir = cfg["file"]["in_dir"]

    keys = ["p", "l"] if args.model == "both" else [args.model]
    for k in keys:
        name, module, tag = VARIANTS[k]
        run_cfg = copy.deepcopy(cfg)
        run_cfg["file"]["out_dir"] = os.path.join(cfg["file"]["out_dir"], tag)
        run_cfg["file"]["model_dir"] = os.path.join(cfg["file"]["model_dir"], tag)
        os.makedirs(run_cfg["file"]["out_dir"], exist_ok=True)
        os.makedirs(run_cfg["file"]["model_dir"], exist_ok=True)

        print(f"\n===== {name} =====")
        for img_id in DEMO_IDS:
            module.train(img_id=img_id, config=run_cfg)
            p, s = evaluate(run_cfg["file"]["out_dir"], data_dir, img_id)
            print(f"  slice {img_id}:  PSNR = {p:6.2f} dB   SSIM = {s:.4f}")


if __name__ == "__main__":
    main()
