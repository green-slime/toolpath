import torch
import config as cfg
import utils
from get_init_pos import get_init_pos
import get_init_pos_scallop

def init_sample_points(sample_size, device='cpu'):
    """
    返回采样点。形状： (sample_size + 1, sample_size + 1, 3)，
    """
    zmin = cfg.zmin
    Ps = utils.make_2d_sample_points(grid_size=sample_size, device=device)
    Ps = torch.cat([Ps, torch.ones((Ps.shape[0], 1), device=device) * zmin], dim=1) # 形状 (sample_size + 1, 3)
    Ps = Ps.view(sample_size + 1, -1, 3).requires_grad_(False)  # 形状 (sample_size + 1, sample_size + 1, 3)，M 是每个采样点的数量
    return Ps

def init_path_points_yzs(path_size, path_num, device='cpu'):
    """
    返回路径点的 y 和 z 分量。形状： (path_num, path_size + 1, 2)，
    每个路径点的 y 和 z 分量分别乘以不同的系数。
    """
    path_points_x = utils.make_1d_sample_points(path_size, device=device).requires_grad_(False)
    path_points_yz = torch.ones((path_num, path_points_x.shape[0], 2), device=device)  # 形状 (path_num, path_size + 1, 2)
    
    y_multipliers = torch.linspace(0.4, 0.6, path_num, device=device).unsqueeze(-1).unsqueeze(-1)  # 形状 (path_num, 1, 1)
    z_multipliers = torch.ones((path_num, 1, 1), device=device) * cfg.z_init # 形状 (path_num, 1, 1)
    
    multipliers = torch.cat([y_multipliers, z_multipliers], dim=-1)  # 形状 (path_num, 1, 2)
    path_points_yz = path_points_yz * multipliers  # 此为路径点的 y 和 z 分量
    
    # 注意这里输出的 path_points_yz 是一个叶子节点
    return path_points_x, path_points_yz.clone().detach().requires_grad_(True)

def get_init_pos_from_surface(device='cpu'):
    """
    从目标曲面获取刀具初始位置。\n
    """
    #centers = get_init_pos(device)
    centers = get_init_pos_scallop.get_resampled_init_pos_tensor(device=device)
    torch.save(centers, 'centers.pt')
    #centers = torch.load('centers.pt', map_location=device)
    return centers[0, :, 0].clone().detach().requires_grad_(False), centers[:, :, 1:].clone().detach().requires_grad_(True)
    
def init(device='cpu'):
    """
    初始化采样点和路径点。
    
    输出：
        采样点 Ps: 形状 (sample_size + 1, sample_size + 1, 3)
        路径点的 x 分量 path_points_x: 形状 (path_size + 1, )
        路径点的 y 和 z 分量 path_points_yz: 形状 (path_num, path_size + 1, 2)
    """
    sample_size = cfg.sample_size
    path_size = cfg.path_size
    path_num = cfg.path_num
    
    Ps = init_sample_points(sample_size, device=device)
    #path_points_x, path_points_yz = init_path_points_yzs(path_size, path_num, device=device)
    path_points_x, path_points_yz = get_init_pos_from_surface(device=device)
    
    return Ps, path_points_x, path_points_yz
