# SA-INR: Scattering-Aware Implicit Neural Representation for CT Metal Artifact Reduction

Minimal demo for the two scattering-aware INR variants from our paper. Each CT
slice is reconstructed by **per-scan self-supervised optimization** directly from
its metal-corrupted sinogram — no training set and no pretrained weights.

- **SA-INR-P** (`sa_inr_p.py`) — physics-constrained Henyey–Greenstein
  single-scatter correction with fixed albedo.
- **SA-INR-L** (`sa_inr_l.py`) — learnable per-scan albedo plus a
  zero-initialized scatter MLP, identical to SA-INR-P at initialization.

Both share a hash-grid INR (tiny-cuda-nn) that predicts the polychromatic
attenuation field, an energy-dependent smoothness regularizer (`model_sc.py`),
and the fan-beam ray geometry in `dataset.py` / `utils.py`.

## Data

Three example 2-D fan-beam slices are bundled under `data/`
(`gt_*`, `mask_*`, `ma_sinogram_*`, plus the shared `fanSensorPos.nii` and
`GE14Spectrum120KVP.mat`). **This release is limited to these three examples.**

## Run

```bash
python run_demo.py              # both variants on all 3 slices
python run_demo.py --model p    # SA-INR-P only
python run_demo.py --model l    # SA-INR-L only
```

Reconstructions are written to `results/<variant>/polyner_<id>.nii`, and PSNR /
SSIM on non-metal regions are printed per slice.

## Requirements

A CUDA GPU is required. Install PyTorch and
[tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn) (its PyTorch bindings),
then:

```bash
pip install -r requirements.txt
```
