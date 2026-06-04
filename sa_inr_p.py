import model_sc, torch, numpy as np, dataset, SimpleITK as sitk, tinycudann as tcnn
from tqdm import tqdm
from torch.utils import data
from scipy import io as scio
from torch.optim import lr_scheduler
from skimage.morphology import erosion, square
from utils import prepare_mask_for_scatter
from scatter_scatter_in import compute_scatter_term

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
     (
      int(SOD - mask.shape[1] / 2), int(SOD - mask.shape[1] / 2) - 1)))).copy()
    # mask = torch.tensor(mask).float().unsqueeze(0).unsqueeze(0).to(device)
    # mask = torch.where(mask == 1, 0.0, 1.0)
    mask_tensor = torch.tensor(mask).float().unsqueeze(0).unsqueeze(0).to(device)  # [1,1,723,723]
    mask_tensor = torch.where(mask_tensor == 1, 0.0, 1.0)  # 非金属为1.0
    # print(mask.shape, "mask")


    # mask_bool = (mask > 0.5)  # 直接判断是否为金属区域
    spectrum_ini = scio.loadmat("./{}/GE14Spectrum120KVP.mat".format(in_path))["GE14Spectrum120KVP"]
    e_1, e_n = (20, 120)
    # spectrum = spectrum[((e_1 - 1)[:e_n], 1)]
    # spectrum_ini = spectrum_ini[(e_1 - 1):e_n, 1]  # ✅ 正确的 Python 索引
    # spectrum = spectrum_ini / np.sum(spectrum_ini)
    spectrum_ini_np = spectrum_ini[(e_1 - 1):e_n, 1]
    spectrum_ave = spectrum_ini_np / np.sum(spectrum_ini_np)
    e_level = len(spectrum_ave)
    spectrum_ini_tensor = torch.tensor(spectrum_ini_np, dtype=torch.float).to(device)

    spectrum = spectrum_ini_tensor / torch.sum(spectrum_ini_tensor)
    spectrum = spectrum.view(1, 1, -1)  # shape [1,1,E]
    # spectrum = torch.tensor(spectrum, dtype=(torch.float)).view(1, 1, -1).to(device)
    dc_loss = torch.nn.L1Loss().to(device)
    # ase_loss = model.Attenuation_Smootion_Over_Energies_Loss(lamb=lamb, mask=mask).to(device)
    ase_loss = model_sc.Attenuation_Smootion_Over_Energies_Loss(lamb=lamb, mask=mask_tensor).to(device)
    network = tcnn.NetworkWithInputEncoding(n_input_dims=2, n_output_dims=e_level, encoding_config=(config["encoding"]),
      network_config=(config["network"])).to(device)
    optimizer = torch.optim.Adam(params=(network.parameters()), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_decay_epoch, gamma=lr_decay_coefficient)
    train_loader = data.DataLoader(
      dataset=dataset.TrainData
      (
          proj_path=proj_path, proj_pos_path=proj_pos_path, SOD=SOD, num_sample_ray=num_sample_ray,
          num_angle=num_angle,
          voxel_size=voxel_size
      ),
      batch_size=batch_size,
      shuffle=True)
    test_loader = data.DataLoader(dataset=dataset.TestData(h=(2 * SOD + 1), w=(2 * SOD + 1)),
      batch_size=1,
      shuffle=False)
    loop_tqdm = tqdm((range(epoch)), leave=False)
    for e in loop_tqdm:
        network.train()
        loss_log = 0
        # for i, (ray, proj, is_metal_region) in enumerate(train_loader):
        for i, (ray, proj) in enumerate(train_loader):
            ray = ray.to(device).float().view(-1, 2)
            proj = proj.to(device).float()
            # is_metal_region = is_metal_region.to(device).bool()

            intensity_pre = network(ray).view(-1, num_sample_ray, 2 * SOD, e_level).float()
            proj_pre = torch.exp(-voxel_size * torch.sum(intensity_pre, dim=2).squeeze(-1))

            is_metal_region = prepare_mask_for_scatter(mask_tensor, intensity_pre)
            
            scatter_pre =  compute_scatter_term(
                intensity_pre=intensity_pre,
                I0=spectrum_ini_tensor,
                voxel_size=voxel_size,
                is_metal_region=is_metal_region,
                g=g_value,          # <-- 加这一行
            )

            actual_det = proj_pre - scatter_pre
            # actual_det = proj_pre
            actual_det = -torch.log(torch.sum((actual_det * spectrum), dim=(-1)).squeeze(-1))
            loss = dc_loss(actual_det, proj.to(actual_det.dtype)) + ase_loss(intensity=intensity_pre, ray=ray)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_log = loss_log + loss.item()
        else:
            scheduler.step()
            loop_tqdm.set_description("Image #{}".format(img_id))
            loop_tqdm.set_postfix(lr=(scheduler.get_last_lr()[0]), loss=(loss_log / len(train_loader)))

        if (e + 1) % save_epoch == 0:
            kx, ky = int(1 + (2 * SOD - h) / 2), int((2 * SOD - w) / 2)
            with torch.no_grad():
                torch.save(network.state_dict(), f"{model_path}/model_{img_id}.pkl")
                
                for i, xy in enumerate(test_loader):
                    xy = xy.to(device).float().view(-1, 2)
                    output = network(xy)  # shape: (N, e_level)
                    e_idx = int(e_level // 2)
                    img_pre_tensor = output[:, e_idx].view(2 * SOD + 1, 2 * SOD + 1)

                    # ✅ 转 float32 NumPy
                    img_pre_np = img_pre_tensor.cpu().detach().numpy().astype(np.float32)

                    # ✅ 裁剪
                    img_pre_np = img_pre_np[kx : kx + h, ky : ky + w]

                    # ✅ 安全 flip
                    if img_pre_np.shape[1] > 0:
                        img_pre_np = np.flip(img_pre_np, axis=1)
                    else:
                        print("⚠️ img_pre_np shape 异常:", img_pre_np.shape)

                # ✅ 保存 NII 图像
                sitk.WriteImage(sitk.GetImageFromArray(img_pre_np), f"{out_path}/polyner_{img_id}.nii")
