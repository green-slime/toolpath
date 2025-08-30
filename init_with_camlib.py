import camlib
import numpy as np
import torch
import resample
import config as cfg
import utils

def init_sample_points(sample_size, device='cpu'):
    """
    返回采样点。形状： (sample_size + 1, sample_size + 1, 3)，
    """
    zmin = cfg.zmin
    Ps = utils.make_2d_sample_points(grid_size=sample_size, device=device)
    Ps = torch.cat([Ps, torch.ones((Ps.shape[0], 1), device=device) * zmin], dim=1) # 形状 (sample_size + 1, 3)
    Ps = Ps.view(sample_size + 1, -1, 3).requires_grad_(False)  # 形状 (sample_size + 1, sample_size + 1, 3)，M 是每个采样点的数量
    return Ps

def get_centers_with_camlib(device='cpu', filename=f'toolpaths/toolpaths_{cfg.ocl_path_num}_{cfg.R}_bdc_{cfg.surface_version}.pkl', need_preprocess=False):
    """
    这里 path_list 是一个嵌套列表，\n
    """
    if(need_preprocess):
        print("需要预处理路径数据。")
        camlib.main(need_mesh=cfg.need_mesh, device=device)
    else:
        print(f"直接加载路径数据 {filename}。")
    path_list = camlib.load_nested_list(filename)
    print(f"路径条数：{len(path_list)}")
    cfg.path_num = len(path_list)  # 更新 cfg.path_num
    new_centers = np.zeros((cfg.path_num, cfg.path_size+1, 3))  # 转换为 NumPy 数组以便使用 resample 函数
    for idx in range(len(path_list)):
        # 需要手动进行半径补正
        new_centers[idx] = resample.resample_toolpath_by_x(np.array(path_list[idx]), cfg.path_size+1) + np.array([0, 0, cfg.R]) # 添加半径补正
    new_centers = torch.tensor(new_centers, device=device, dtype=torch.float32)  # 转换回张量并返回
    #print(new_centers.shape)
    return new_centers

def init_with_camlib(device='cpu', filename=f'toolpaths/toolpaths_{cfg.ocl_path_num}_{cfg.R}_bdc_{cfg.surface_version}.pkl', use_x=False):
    """
    从 camlib 中加载路径点，并将其转换为初始位置。
    """
    centers = get_centers_with_camlib(device=device, filename=filename, need_preprocess=cfg.need_toolpath_preprocess)
    # 将 centers 保存到文件
    torch.save(centers, f'centers/centers_{cfg.ocl_path_num}_{cfg.R}_bdc_{cfg.surface_version}.pt')
    print(f"初始位置已保存至 centers/centers_{cfg.ocl_path_num}_{cfg.R}_bdc_{cfg.surface_version}.pt。")
    Ps = init_sample_points(cfg.sample_size, device=device)
    if use_x:
        return Ps, centers.clone().detach().requires_grad_(True)
    else:
        return Ps, centers[:, :, 1:].clone().detach().requires_grad_(True)


if __name__ == "__main__":

    device = utils.cuda_init(0)
    init_with_camlib(device=device, filename='toolpaths.pkl')