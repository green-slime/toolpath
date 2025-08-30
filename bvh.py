import torch
import utils

def make_x_grid(path_size, sample_size, R, device='cpu'):
    """
    因为路径点和采样点在 x 方向固定，可以预计算可能相交的全部路径点。 \n
    输入：
        path_size (int): 路径点数量
        sample_size (int): 采样数量
    输出：
        x_grid (torch.tensor): 可能相交的路径点索引 i, j 的列表，形状 (sample_size + 1, 2)
    """
    # Pi.x + R <= x <= Pj.x - R
    x_k = utils.make_1d_sample_points(grid_size=sample_size, device=device)  # 采样点 x 坐标
    x_grid = torch.zeros((sample_size + 1, 2), dtype=torch.int32, requires_grad=False)
    x_grid[:,0] = torch.maximum(torch.tensor([0], device=device), torch.floor(path_size*(x_k - R)).squeeze()) # i >= 0
    x_grid[:,1] = torch.minimum(torch.tensor([path_size], device=device), (torch.ceil(path_size*(x_k + R)) + 1).squeeze()) # j <= path_size

    return x_grid.long()

def get_possible_paths(path_yzs, x_grid, k):
    """
    从 x_grid 中获取可能相交的路径点。 \n
    输入：
        path_yzs (torch.tensor): 路径点的 y,z 分量，形状 (path_num, path_size + 1, 2)
        x_grid (torch.tensor): 可能相交的路径点索引 i, j 的列表，形状 (sample_size + 1, 2)
        k (int): 采样点 x 方向上的索引
    输出：
        对应的路径点 path_yzs，形状 (path_num, j-i+1, 2) \n
        对应的 x 坐标 path_xs，形状 (j-i+1, )
    """
    i, j = x_grid[k]
    #print(i, j)
    path_xs = torch.arange(i, j+1, device=path_yzs.device) / (path_yzs.shape[1] - 1)
    #print(path_xs, path_xs.shape)
    return path_xs, path_yzs[:,i:j+1,:]

def make_path_points(path_yzs, x_grid, k):
    """
    假设有 path_num 条路径，每条路径的路径点数量以及位置由 get_path_size 函数返回。 \n
    我们现在直接生成这些 path_points，以供 intersect.py 使用。 \n
    输出：
        path_points (torch.tensor): 路径点的 x,y,z 坐标，形状 (path_num, j-i+1, 3) \n
    """
    path_xs, path_yz_possible = get_possible_paths(path_yzs, x_grid, k)
    path_num = path_yz_possible.shape[0]
    path_xs_expanded = path_xs.unsqueeze(0).unsqueeze(-1).expand(path_num,path_yz_possible.shape[1], 1) # 形状 (path_num, j-i+1, 1)
    path_points = torch.cat([path_xs_expanded, path_yz_possible], dim=-1)
    #print(path_points.shape)
    return path_points
    
def batch_make_path_points(path_yzs, x_grid, k_list:list):
    """
    现在 k 不是一个整数，而是一个列表，表示多列采样点的 x 方向索引。 \n
    注意 x_k 对应 (i_k, j_k)，即每个 k 对应一对 (i, j)。 \n
    那么，[x_k1, ..., x_kn] 对应 [(i_k1, j_k1), ..., (i_kn, j_kn)]。 \n
    实际上完全可以直接考虑 (i_k1, j_kn) \n
    输出：
        path_points (torch.tensor): 路径点的 x,y,z 坐标，形状 (path_num, j-i+1, 3) \n
    """
    k_list.sort()
    k_min = k_list[0]
    k_max = k_list[-1]
    
    i, _ = x_grid[k_min]
    _, j = x_grid[k_max]
    
    path_xs = torch.arange(i, j+1, device=path_yzs.device) / (path_yzs.shape[1] - 1)
    path_yz_possible = path_yzs[:, i:j+1, :]
    path_num = path_yz_possible.shape[0]
    path_xs_expanded = path_xs.unsqueeze(0).unsqueeze(-1).expand(path_num,path_yz_possible.shape[1], 1) # 形状 (path_num, j-i+1, 1)
    path_points = torch.cat([path_xs_expanded, path_yz_possible], dim=-1)
    #print(path_points.shape)
    return path_points, i, j