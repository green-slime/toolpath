import torch
import config as cfg
from old_code.NURBS import NURBS
import utils
import resample
import numpy as np

def get_centers(surface_points, normals, R):
    centers = surface_points + normals * R
    return centers

def get_nurbs_data(sample_points, device='cpu'):
    old_project_path = '/data/wzr/2025'
    control_points1 = torch.load(f'{old_project_path}/{cfg.control_points_name}').to(device)
    wij = torch.load(f'{old_project_path}/{cfg.wij_name}').to(device)
    #control_points1 = torch.ones((100, 100)).to(device)
    #wij = torch.ones((100, 100)).to(device)

    bsurface = NURBS(control_points1, degree_u=3, degree_v=3, sample_points=sample_points)
    #print(sample_points)
    heights, normals = bsurface.evaluate_batch(sample_points, control_points1, wij, batch_size=100000)
    #heights, normals, k1, k2 = bsurface.evaluate_curvature_batch(sample_points, control_points1, wij, batch_size=100000)
    surface_points = torch.cat((sample_points, heights.unsqueeze(-1)), dim=-1)  # [N, 3]
    return surface_points, normals

def generate_sample_points(x_num=cfg.path_size+1, y_num=cfg.path_num, device='cpu'):
    """
    需要生成 y 方向 path_num 个点， x 方向 path_size + 1 个点，
    """
    xs = torch.linspace(0., 1., x_num)  # x 方向的采样点
    ys = torch.linspace(0., 1., y_num)  # y 方向的采样点
    x_grid, y_grid = torch.meshgrid(xs, ys, indexing='ij')
    sample_points = torch.stack((x_grid.flatten(), y_grid.flatten()), dim=-1)  # [path_num * (path_size + 1), 2]
    return sample_points.to(device)

def reshape_data(centers, x_num=cfg.path_size + 1, y_num=cfg.path_num):
    """
    上面得到的 surface_points 是 [N, 3] 的张量，并按先变 y 再变 x 的顺序排列
    现在要将其 reshape 成 [path_num, path_size + 1, 3] 的张量，
    """
    centers = centers.reshape(x_num, y_num, 3)
    centers = centers.transpose(0, 1)  # 转置为 [path_num, path_size + 1, 3]
    return centers

def make_centers_to_fixed_xs(centers, device='cpu'):
    """
    上述计算后得到球心的精确位置，然而我们的模型下，所有球心的 x 坐标必须固定.
    输入：
        centers: [path_num, path_size + 1, 3] 的张量
    """
    x_s = torch.linspace(0, 1, cfg.path_size + 1).to(device)  # x 坐标
    x_s = x_s[None, :, None].expand(centers.shape[0], -1, 1) # x_s 的形状变为 [path_num, path_size + 1, 1]
    new_centers = centers.clone()
    new_centers[:, :-1, 1:] = (centers[:, 1:, 1:]-centers[:, :-1, 1:]) / (centers[:, 1:, 0:1]-centers[:, :-1, 0:1])* (x_s[:, :-1, 0:1]-centers[:, :-1, 0:1]) + centers[:, :-1, 1:]  #    
    new_centers[:, -1, 1:] = (centers[:, -1, 1:]-centers[:, -2, 1:]) / (centers[:, -1, 0:1]-centers[:, -2, 0:1])* (x_s[:, -1, 0:1]-centers[:, -2, 0:1]) + centers[:, -2, 1:] # 最后一列的坐标特殊处理
    new_centers[:, :, 0] = x_s[:, :, 0]  # 固定 x 坐标
    return new_centers

def make_centers_to_fixed_xs_by_resample(centers, device='cpu'):
    """
    利用 resample_toolpath_by_x 函数对 centers 进行重采样
    输入：
        centers: [path_num, path_size + 1, 3] 的张量；或其他形状的张量
    输出：
        new_centers: [path_num, path_size + 1, 3] 的张量，x 坐标固定
    """
    #print(centers.shape)
    new_centers = np.zeros((cfg.path_num, cfg.path_size+1, 3))  # 转换为 NumPy 数组以便使用 resample 函数
    for idx in range(centers.shape[0]):
        new_centers[idx] = resample.resample_toolpath_by_x(centers[idx].cpu().numpy(), cfg.path_size+1)
    new_centers = torch.tensor(new_centers, device=device, dtype=torch.float32)  # 转换回张量并返回
    #print(new_centers.shape)
    return new_centers
    
    
def get_init_pos(device='cpu'):    
    """
    返回从目标曲面获取的刀具初始位置。\n
    刀具初始位置是一个形状为 [path_num, path_size + 1, 3] 的张量，\n
    且其 x 坐标已固定。
    """
    #x_num = cfg.path_size + 1
    #y_num = cfg.path_num
    x_num = cfg.path_size + 1
    y_num = cfg.path_num
    sample_points = generate_sample_points(x_num, y_num, device=device)
    #print(sample_points, sample_points.shape)
    surface_points, normals = get_nurbs_data(sample_points, device)
    centers = get_centers(surface_points, normals, cfg.R)
    centers = reshape_data(centers, x_num, y_num)
    new_centers = make_centers_to_fixed_xs_by_resample(centers, device)
    return new_centers
