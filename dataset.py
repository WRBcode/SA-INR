import utils, numpy as np, SimpleITK as sitk
from torch.utils import data

class TrainData(data.Dataset):

    def __init__(self, proj_path, proj_pos_path, num_sample_ray, num_angle, SOD, voxel_size):
        self.num_angle = num_angle
        self.num_sample_ray = num_sample_ray
        self.SOD = SOD
        self.voxel_size = voxel_size
        self.angles = np.linspace(0.0, 360.0, num=(self.num_angle), endpoint=False)
        self.proj_pos = sitk.GetArrayFromImage(sitk.ReadImage(proj_pos_path)).reshape(-1)
        self.num_det = len(self.proj_pos)
        self.proj = sitk.GetArrayFromImage(sitk.ReadImage(proj_path))
        self.rays = utils.fan_beam_ray(self.proj_pos, self.SOD)
        self.index_max = self.num_det - self.num_sample_ray

        self.proj_path = proj_path  # 保存路径，用于调试
        self.proj_pos_path = proj_pos_path  # 保存路径，用于调试

    def __getitem__(self, item):
        ang = self.angles[item]
        proj = self.proj[item].reshape(-1)
        index = np.random.randint(0, (self.index_max), size=1)[0]
        ray_sample = self.rays[index : index + self.num_sample_ray]
        proj_sample = proj[index : index + self.num_sample_ray]

        # 检查 ray_sample 和 proj_sample 的形状是否一致
        if ray_sample.shape[0] != proj_sample.shape[0]:
            print(f"⚠️ Mismatch between ray_sample and proj_sample sizes at index {item}.")
            print(f"  ray_sample shape: {ray_sample.shape}")
            print(f"  proj_sample shape: {proj_sample.shape}")
            return None  # 返回空数据


        # 检查 ray_sample 和 proj_sample 的尺寸
        if ray_sample.shape[0] == 0 or proj_sample.shape[0] == 0:
            print(f"⚠️ Empty data found at index {item}. Sample path info: ")
            print(f"  proj_path: {self.proj_path}")
            print(f"  proj_pos_path: {self.proj_pos_path}")
            print(f"  Angles: {self.angles}")
            return None  # 如果数据为空，跳过当前样本

        # 确保 ray_sample 和 proj_sample 的尺寸一致
        if ray_sample.shape[0] != proj_sample.shape[0]:
            print(f"⚠️ Mismatch between ray_sample and proj_sample sizes at index {item}. Sample path info: ")
            print(f"  proj_path: {self.proj_path}")
            print(f"  proj_pos_path: {self.proj_pos_path}")
            print(f"  Angles: {self.angles}")
            return None  # 如果尺寸不一致，跳过当前样本

        ray_sample = utils.rotate_ray(xy=ray_sample, angle=ang)
        return (ray_sample, proj_sample)

    def __len__(self):
        return self.num_angle


class TestData(data.Dataset):

    def __init__(self, h, w):
        self.h, self.w = h, w
        self.xy = utils.grid_coordinate(h=(self.h), w=(self.w)).reshape(1, int(h * w), 2)

    def __getitem__(self, item):
        return self.xy[item]

    def __len__(self):
        return 1

