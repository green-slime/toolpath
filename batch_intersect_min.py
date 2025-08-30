import torch
from tqdm import tqdm
import bvh
import intersect_extension
import config as cfg
import render
import prepare_for_render
from time import time

Z_INF = 1e12  # 用于表示无效的高度

def batch_intersect_for_render(Ps, R, sample_size, x_grid, path_yzs, sobel_result, infolist, batch_size=1, device='cuda', MAX_HEIGHT=1.0):
    """
    功能与下同，但由渲染 loss 监督。
    """
    loss = 0.0
    grad = torch.zeros_like(path_yzs, device=device)
    heights = []
    normals = []
    for k_start in range(0, sample_size + 1, batch_size):
        k_end = min(k_start + batch_size, sample_size + 1)
        k_batch = list(range(k_start, k_end))  # [k_start, k_start+1, ..., k_end-1]
        # 批量获取采样点
        Ps_k = Ps[k_batch].reshape(-1 ,3)  # 形状 (batch_size * (sample_size + 1), 3)
        # 取出 batch_size 列采样点，它们的 x 坐标是 x_{k_start} 到 x_{k_end - 1}
        path_points, _, _ = bvh.batch_make_path_points(path_yzs, x_grid, k_batch)  # 获取可能相交的包络面
        height, normal = intersect_extension.intersect(Ps_k.contiguous(), path_points.contiguous(), R, MAX_HEIGHT, False)  
        heights.append(height)
        normals.append(normal)
    # test: backward()
    #exit()
    heights = torch.cat(heights, dim=0)
    normals = torch.cat(normals, dim=0)
    with torch.no_grad():
        receiver_points = render.trace_rays_through_surface(torch.cat([Ps[:, :, :2].reshape(-1, 2), heights.unsqueeze(-1)], dim=-1), normals, cfg.z_of_receiver, n1=cfg.n1, n2=cfg.n2)  # 形状 ((sample_size + 1) * (sample_size + 1), 2)
        points_diff, render_diff = prepare_for_render.get_diff(receiver_points, infolist, sobel_result)  # 计算渲染损失
        points_diff = points_diff.reshape(sample_size + 1, sample_size + 1, 2)  # 形状 ((sample_size + 1) * (sample_size + 1), 2)

    for k_start in range(0, sample_size + 1, batch_size):
        k_end = min(k_start + batch_size, sample_size + 1)
        k_batch = list(range(k_start, k_end))  # [k_start, k_start+1, ..., k_end-1]
        # 批量获取采样点
        Ps_k = Ps[k_batch].reshape(-1 ,3)  # 形状 (batch_size * (sample_size + 1), 3)
        # 取出 batch_size 列采样点，它们的 x 坐标是 x_{k_start} 到 x_{k_end - 1}
        path_points_k, i, j = bvh.batch_make_path_points(path_yzs, x_grid, k_batch)  # 获取可能相交的包络面
        points_diff_k = points_diff[k_batch].reshape(-1, 2)  # 形状 (batch_size * (sample_size + 1), 2)
        grad_render_to_path_points_k = intersect_extension.intersect_with_grad_for_render(Ps_k.contiguous(), path_points_k.contiguous(), points_diff_k.contiguous(), R, MAX_HEIGHT, cfg.n1, cfg.n2, cfg.z_of_receiver)[0]  # 计算梯度
        grad[:, i:j+1, :] += grad_render_to_path_points_k  
    loss = torch.norm(render_diff, p=2)  # 累加损失
    return loss, grad, receiver_points
        

def batch_intersect_with_cuda(Ps, R, sample_size, x_grid, path_yzs, batch_size = 1, device='cuda', alpha=1000.0, MAX_HEIGHT=1.0, target_pos=None, need_dirs=False):
    """
    在诸 x_k 中 for 循环，每个 x_k 对应一列采样点 Ps_k，通过 bvh 确定可能相交的所有包络面，对一列采样点和相关的包络面进行射线扫掠，向量化计算每个采样点的高度和法向量。
    
    输入：
        Ps: 采样点，形状 (sample_size+1, sample_size+1, 3)，每个采样点的 [x, y, z] 坐标
        batch_size: 一次处理多少列采样点
    输出：
        heights: 每个采样点的高度，形状 ((sample_size+1)**2,)
        normals: 每个采样点的法向量，形状 ((sample_size+1)**2, 3)
    """ 
    # 添加 z 分量为 0
    heights = []
    normals = []
    dirs = []
    loss = 0.0
    grad = torch.zeros_like(path_yzs)
    if target_pos is not None:
        target_pos = target_pos.reshape(Ps.shape[0], Ps.shape[1], 2)
        target_pos = torch.cat([target_pos, torch.ones_like(target_pos[:, :, :1]) * cfg.z_of_receiver], dim=-1)  # 添加 z 分量为 z_height
    for k_start in range(0, sample_size + 1, batch_size):
        k_end = min(k_start + batch_size, sample_size + 1)
        k_batch = list(range(k_start, k_end))  # [k_start, k_start+1, ..., k_end-1]
        # 批量获取采样点
        Ps_k = Ps[k_batch].reshape(-1 ,3)  # 形状 (batch_size * (sample_size + 1), 3)
  
        # 取出 batch_size 列采样点，它们的 x 坐标是 x_{k_start} 到 x_{k_end - 1}
        path_points, i, j = bvh.batch_make_path_points(path_yzs, x_grid, k_batch)  # 获取可能相交的包络面  
        # 调用 pybind 插件时不能使用关键字参数
        if target_pos is None:
            height2, normal2, dir2 = intersect_extension.intersect(Ps_k.contiguous(), path_points.contiguous(), R, MAX_HEIGHT, True)
            dirs.append(dir2)
        else:
            target_pos_k = target_pos[k_batch].reshape(-1 ,3)
            height2, normal2, loss2, grad2 = intersect_extension.intersect_with_grad(Ps_k.contiguous(), path_points.contiguous(), target_pos_k.contiguous(), R, MAX_HEIGHT, cfg.n1, cfg.n2, cfg.z_of_receiver)
            grad[:, i:j+1, :] += grad2[:,:,1:]  # 累加梯度, x 方向不需要
            loss += loss2.sum()  # 累加损失
        heights.append(height2)
        normals.append(normal2)       
    # test: backward()
    #exit()
    heights = torch.cat(heights, dim=0)
    normals = torch.cat(normals, dim=0)
    if dirs:
        dirs = torch.cat(dirs, dim=0)
    if(need_dirs):
        return heights, normals, dirs, loss, grad
    else:
        return heights, normals, loss, grad
    
def batch_intersect_for_normals(Ps, R, sample_size, x_grid, path_yzs, batch_size = 1, device='cuda', alpha=1000.0, MAX_HEIGHT=1.0, target_normals=None, need_dirs=False):
    """
    在诸 x_k 中 for 循环，每个 x_k 对应一列采样点 Ps_k，通过 bvh 确定可能相交的所有包络面，对一列采样点和相关的包络面进行射线扫掠，向量化计算每个采样点的高度和法向量。
    
    输入：
        Ps: 采样点，形状 (sample_size+1, sample_size+1, 3)，每个采样点的 [x, y, z] 坐标
        batch_size: 一次处理多少列采样点
    输出：
        heights: 每个采样点的高度，形状 ((sample_size+1)**2,)
        normals: 每个采样点的法向量，形状 ((sample_size+1)**2, 3)
    """ 
    # 添加 z 分量为 0
    heights = []
    normals = []
    dirs = []
    loss = 0.0
    grad = torch.zeros_like(path_yzs)
    if target_normals is not None:
        target_normals = target_normals.reshape(Ps.shape[0], Ps.shape[1], 3)
    for k_start in range(0, sample_size + 1, batch_size):
        k_end = min(k_start + batch_size, sample_size + 1)
        k_batch = list(range(k_start, k_end))  # [k_start, k_start+1, ..., k_end-1]
        # 批量获取采样点
        Ps_k = Ps[k_batch].reshape(-1 ,3)  # 形状 (batch_size * (sample_size + 1), 3)
  
        # 取出 batch_size 列采样点，它们的 x 坐标是 x_{k_start} 到 x_{k_end - 1}
        path_points, i, j = bvh.batch_make_path_points(path_yzs, x_grid, k_batch)  # 获取可能相交的包络面  
        # 调用 pybind 插件时不能使用关键字参数
        if target_normals is None:
            height2, normal2, dir2 = intersect_extension.intersect(Ps_k.contiguous(), path_points.contiguous(), R, MAX_HEIGHT, True)
            dirs.append(dir2)
        else:
            target_normals_k = target_normals[k_batch].reshape(-1 ,3)
            height2, normal2, loss2, grad2 = intersect_extension.intersect_with_normal_grad(Ps_k.contiguous(), path_points.contiguous(), target_normals_k.contiguous(), R, MAX_HEIGHT, cfg.n1, cfg.n2, cfg.z_of_receiver)
            grad[:, i:j+1, :] += grad2[:,:,1:]  # 累加梯度, x 方向不需要
            loss += loss2.sum()  # 累加损失
        heights.append(height2)
        normals.append(normal2)       
    # test: backward()
    #exit()
    heights = torch.cat(heights, dim=0)
    normals = torch.cat(normals, dim=0)
    if dirs:
        dirs = torch.cat(dirs, dim=0)
    if(need_dirs):
        return heights, normals, dirs, loss, grad
    else:
        return heights, normals, loss, grad
    
import new_intersect_extension    
def intersect_for_height(Ps, R, sample_size, path_yzs, MAX_HEIGHT=1.0, target_height=None):
    """
    新的 cuda 代码下不需要 batch 处理，直接对所有采样点进行处理。
    """
    gouge_weight = 5.0
    from time import time
    start_time = time()
    if target_height is not None:
        # 生成 x 分量并拼接到 path_yzs 上
        path_num, path_len = path_yzs.shape[:2]
        x_coords = torch.linspace(0, 1, path_len, device=path_yzs.device).unsqueeze(0).expand(path_num, -1).unsqueeze(-1)
        path_points = torch.cat([x_coords, path_yzs], dim=-1)  # 形状变为 [path_num, path_len, 3]
        height2, normal2, loss2, grad2 = new_intersect_extension.intersect_with_height_grad(Ps.contiguous(), path_points.contiguous(), R, MAX_HEIGHT, target_height.contiguous(), gouge_weight)
        loss = loss2.sum()
        print(f"求交 time: {time() - start_time:.4f} s")
        #print(f"target height 第一行: {target_height.reshape(sample_size+1, sample_size+1)[:,0]}")
        #print(f"height2 第一行：{height2.reshape(sample_size+1, sample_size+1)[:,0]}")
        return height2, normal2, loss, grad2[:,:,1:]
    else:
        raise NotImplementedError("需要提供 target_height 进行高度监督。")
    
def intersect_new(Ps, R, sample_size, path_yzs, MAX_HEIGHT=1.0):
    """
    不用 batch
    """
    pass

    
def batch_intersect_for_height(Ps, R, sample_size, x_grid, path_yzs, batch_size = 1, device='cuda', alpha=1000.0, MAX_HEIGHT=1.0, target_height=None, need_dirs=False):
    """
    在诸 x_k 中 for 循环，每个 x_k 对应一列采样点 Ps_k，通过 bvh 确定可能相交的所有包络面，对一列采样点和相关的包络面进行射线扫掠，向量化计算每个采样点的高度和法向量。
    
    输入：
        Ps: 采样点，形状 (sample_size+1, sample_size+1, 3)，每个采样点的 [x, y, z] 坐标
        batch_size: 一次处理多少列采样点
        target_height: ((N**2,))
    输出：
        heights: 每个采样点的高度，形状 ((sample_size+1)**2,)
        normals: 每个采样点的法向量，形状 ((sample_size+1)**2, 3)
    """ 
    # 添加 z 分量为 0
    heights = []
    heights_for_check = []
    normals = []
    dirs = []
    loss = 0.0
    grad = torch.zeros_like(path_yzs)
    if target_height is not None:
        target_height = target_height.reshape(Ps.shape[0], Ps.shape[1])
    for k_start in range(0, sample_size + 1, batch_size):
        k_end = min(k_start + batch_size, sample_size + 1)
        k_batch = list(range(k_start, k_end))  # [k_start, k_start+1, ..., k_end-1]
        # 批量获取采样点
        Ps_k = Ps[k_batch].reshape(-1 ,3)  # 形状 (batch_size * (sample_size + 1), 3)
  
        # 取出 batch_size 列采样点，它们的 x 坐标是 x_{k_start} 到 x_{k_end - 1}
        path_points, i, j = bvh.batch_make_path_points(path_yzs, x_grid, k_batch)  # 获取可能相交的包络面  
        # 调用 pybind 插件时不能使用关键字参数
        if target_height is None:
            height2, normal2, dir2 = intersect_extension.intersect(Ps_k.contiguous(), path_points.contiguous(), R, MAX_HEIGHT, True)
            dirs.append(dir2)
        else:
            time1 = time()
            target_height_k = target_height[k_batch].reshape(-1 ,)
            height2, normal2, loss2, grad2, heights_for_check2 = intersect_extension.intersect_with_height_grad(Ps_k.contiguous(), path_points.contiguous(), R, MAX_HEIGHT, target_height_k.contiguous())
            grad[:, i:j+1, :] += grad2[:,:,1:]  # 累加梯度, x 方向不需要
            loss += loss2.sum()  # 累加损失
            heights_for_check.append(heights_for_check2)
            print(f"求交 time: {time() - time1:.4f} s")

        heights.append(height2)
        normals.append(normal2)
        
    # test: backward()
    #exit()
    heights = torch.cat(heights, dim=0)
    normals = torch.cat(normals, dim=0)
    heights_for_check = torch.cat(heights_for_check, dim=0)
    high_count = (heights > 0.2).sum().item()
    if(high_count > 0):
        print(f"警告：{high_count} 个采样点高度大于 0.2，出现无交情形！")
    if(loss > 1.0):
        print(f"警告：损失过大，当前损失为 {loss.item()}！")
        # 打印高度差距较大的点的详细信息
        height_diff = heights.reshape(-1) - heights_for_check.reshape(-1)
        #loss_mask = height_diff > 0.0  
        #loss_we_say = (torch.pow(height_diff[loss_mask],2)).sum().item()+(torch.pow(height_diff[~loss_mask],2)*5.0).sum().item()  # 计算损失
        #print(f"loss we say: {loss_we_say:.4f}, loss: {loss.item():.4f}")
        # 将height_diff保存到文件
        height_diff_reshaped = height_diff.reshape(sample_size + 1, sample_size + 1)
        with open('height_diff_debug.txt', 'w') as f:
            f.write(f"Height diff shape: {height_diff_reshaped.shape}\n")
            f.write(f"Max diff: {height_diff.max().item():.6f}\n")
            f.write(f"Min diff: {height_diff.min().item():.6f}\n")
            f.write(f"Mean diff: {height_diff.mean().item():.6f}\n")
            f.write(f"Std diff: {height_diff.std().item():.6f}\n\n")
            f.write("Height diff matrix:\n")
            for i in range(sample_size + 1):
                for j in range(sample_size + 1):
                    f.write(f"{height_diff_reshaped[i, j].item():.6f}\t")
                f.write("\n")
        large_diff_mask = height_diff > 0.001  # 阈值可调整
        large_diff_indices = torch.where(large_diff_mask)[0]
        if len(large_diff_indices) > 0:
            print(f"发现 {len(large_diff_indices)} 个高度差距较大的点:")
            for idx in large_diff_indices[:10]:  # 只显示前10个
                i, j = idx // (sample_size + 1), idx % (sample_size + 1)
                print(f"  点({i},{j}): 计算高度={heights.reshape(-1)[idx]:.4f}, 目标高度={target_height.reshape(-1)[idx]:.4f}, 差值={height_diff[idx]:.4f}")
    if(dirs):
        dirs = torch.cat(dirs, dim=0)
    if(need_dirs):
        return heights, normals, dirs, loss, grad
    else:
        return heights, normals, loss, grad
    
def batch_intersect(Ps, R, sample_size, x_grid, path_yzs, batch_size = 1, device='cuda', alpha=1000.0, MAX_HEIGHT=1.0):   
    """
    在诸 x_k 中 for 循环，每个 x_k 对应一列采样点 Ps_k，通过 bvh 确定可能相交的所有包络面，对一列采样点和相关的包络面进行射线扫掠，向量化计算每个采样点的高度和法向量。
    
    输入：
        Ps: 采样点，形状 (sample_size+1, sample_size+1, 3)，每个采样点的 [x, y, z] 坐标
        batch_size: 一次处理多少列采样点
    输出：
        heights: 每个采样点的高度，形状 ((sample_size+1)**2,)
        normals: 每个采样点的法向量，形状 ((sample_size+1)**2, 3)
    """ 
    # 添加 z 分量为 0
    heights = []
    normals = []
    for k_start in range(0, sample_size + 1, batch_size):
        k_end = min(k_start + batch_size, sample_size + 1)
        k_batch = list(range(k_start, k_end))  # [k_start, k_start+1, ..., k_end-1]
        # 批量获取采样点
        Ps_k = Ps[k_batch].reshape(-1 ,3)  # 形状 (batch_size * (sample_size + 1), 3)
        # 取出 batch_size 列采样点，它们的 x 坐标是 x_{k_start} 到 x_{k_end - 1}
        path_points, _, _ = bvh.batch_make_path_points(path_yzs, x_grid, k_batch)  # 获取可能相交的包络面
        height, normal= ray_sweep_intersection(Ps_k, R, path_points, device=device, alpha=alpha, MAX_HEIGHT=MAX_HEIGHT)
        # 调用 pybind 插件时不能使用关键字参数
        #height2, normal2= intersect_extension.intersect(Ps_k.contiguous(), path_points.contiguous(), R, MAX_HEIGHT)
        #height2, normal2, grad_s, grad_n= intersect_extension.intersect_with_grad(Ps_k.contiguous(), path_points.contiguous(), R, MAX_HEIGHT)
        #print(torch.max(torch.abs(height-height2)))
        #print(normal, normal2)
        #print(height.requires_grad)
        #print(path_yzs.shape, path_points.shape)
        #height2, normal2, loss2, grad2 = intersect_extension.intersect_with_grad(Ps_k.contiguous(), path_points.contiguous(), target_pos_k.contiguous(), R, MAX_HEIGHT, cfg.n1, cfg.n2, cfg.z_of_receiver)
        #height.sum().backward()  # 计算梯度
        
        heights.append(height)
        normals.append(normal)
    # test: backward()
    #exit()
    heights = torch.cat(heights, dim=0)
    normals = torch.cat(normals, dim=0)

    return heights, normals


def ray_sweep_intersection(P, R, path_points, device='cuda', alpha=1000.0, MAX_HEIGHT=1.0):
    """
    计算多条射线与球头刀扫掠体的交点，返回最近交点的高度和法向量。
    参数：
        P: 射线起点，形状 (N, 3)，[Px, Py, Pz]，固定常量
        R: 刀具半径，标量
        path_points: 路径点，形状 (path_num, j-i+1, 3)，每个路径点的 [x, y, z] 坐标
        device: 计算设备，'cuda' 或 'cpu'
        alpha: softmin 的锐化系数
        MAX_HEIGHT: 默认最大高度
    返回：
        height: 最近交点的 Z 坐标，形状 (N,)
        normal: 最近交点的法向量，形状 (N, 3)
    """
    # 确保输入为 PyTorch 张量
    R = torch.tensor(R, dtype=torch.float32, device=device, requires_grad=False)

    N = P.shape[0]
    M_tubular = path_points.shape[0] * (path_points.shape[1] - 1)  # 管状段数
    M_spherical = path_points.shape[0] * path_points.shape[1]  # 球面点数

    # 初始化交点集合
    s_values = []
    normals = []
    valid_flats = []

    # --- 管状表面交点 ---
    Pi = path_points[:, :-1, :].reshape(M_tubular, 3)  # 形状 (M_tubular, 3)
    Pj = path_points[:, 1:, :].reshape(M_tubular, 3)   # 形状 (M_tubular, 3)

    # 管状表面边界框
    box_min = torch.minimum(Pi, Pj) - R  # 形状 (M_tubular, 3)
    box_max = torch.maximum(Pi, Pj) + R  # 形状 (M_tubular, 3)

    # 检查射线与边界框的 XY 交点
    mask_xy = (P[:, 0:1] >= box_min[:, 0]) & (P[:, 0:1] <= box_max[:, 0]) & \
              (P[:, 1:2] >= box_min[:, 1]) & (P[:, 1:2] <= box_max[:, 1])  # 形状 (N, M_tubular)

    if mask_xy.any():
        # 扩展维度
        Pi_exp = Pi.unsqueeze(0).expand(N, M_tubular, 3)  # 形状 (N, M_tubular, 3)
        Pj_exp = Pj.unsqueeze(0).expand(N, M_tubular, 3)  # 形状 (N, M_tubular, 3)
        P_exp = P[:, None, :].expand(N, M_tubular, 3)  # 形状 (N, M_tubular, 3)

        # 应用掩码，使用 inf 填充无效段
        valid_seg = mask_xy  # 形状 (N, M_tubular)
        inf_fill = torch.full_like(Pi_exp, float('inf'))  # 形状 (N, M_tubular, 3)
        Pi_valid = torch.where(valid_seg.unsqueeze(-1), Pi_exp, inf_fill)  # 形状 (N, M_tubular, 3)
        Pj_valid = torch.where(valid_seg.unsqueeze(-1), Pj_exp, inf_fill)  # 形状 (N, M_tubular, 3)

        Di = Pj_valid - Pi_valid  # 形状 (N, M_tubular, 3)
        Di = torch.where(torch.isfinite(Di), Di, torch.zeros_like(Di))  # 无效 Di 设为零
        Qi = P_exp - Pi_valid  # 形状 (N, M_tubular, 3)
        Qi = torch.where(torch.isfinite(Qi), Qi, torch.zeros_like(Qi))  # 无效 Qi 设为零
        Di_norm2 = torch.sum(Di**2, dim=-1)  # 形状 (N, M_tubular)

        # 管状表面交点
        a = Di_norm2 - Di[:, :, 2]**2  # 形状 (N, M_tubular)
        b = 2 * (-torch.sum(Qi * Di, dim=-1) * Di[:, :, 2] + Qi[:, :, 2] * Di_norm2)  # 形状 (N, M_tubular)
        c = Di_norm2 * (torch.sum(Qi**2, dim=-1) - R**2) - torch.sum(Qi * Di, dim=-1)**2  # 形状 (N, M_tubular)

        # 判别式
        discriminant = b**2 - 4 * a * c  # 形状 (N, M_tubular)
        valid = (discriminant > 0) & valid_seg  # 形状 (N, M_tubular)

        # 解二次方程
        sqrt_d = torch.sqrt(torch.where(valid, discriminant, torch.zeros_like(discriminant)))  # 形状 (N, M_tubular)
        s1 = torch.where(valid, (-b + sqrt_d) / (2 * a), Z_INF)  # 形状 (N, M_tubular)
        s2 = torch.where(valid, (-b - sqrt_d) / (2 * a), Z_INF)  # 形状 (N, M_tubular)

        # 合并 s1 和 s2
        s = torch.stack([s1, s2], dim=-1)  # 形状 (N, M_tubular, 2)
        valid_s = (s >= 0) & valid.unsqueeze(-1)  # 形状 (N, M_tubular, 2)

        # 计算 t
        t = (torch.sum(Qi * Di, dim=-1, keepdim=True) + s * Di[:, :, 2:3]) / (Di_norm2.unsqueeze(-1))  # 形状 (N, M_tubular, 2)
        valid_t = (t >= 0) & (t <= 1) & valid_s  # 形状 (N, M_tubular, 2)

        # 展平
        s_flat = s.view(N, M_tubular*2)  # 形状 (N, M_tubular*2)
        valid_flat = valid_t.view(N, M_tubular*2)  # 形状 (N, M_tubular*2)
        t_flat = t.view(N, M_tubular*2)  # 形状 (N, M_tubular*2)

        # 计算交点和法向量
        intersect = P[:, None, :] + s_flat.unsqueeze(-1) * torch.tensor([0, 0, 1], device=device)  # 形状 (N, M_tubular*2, 3)
        Di_flat = Di.unsqueeze(-2).expand(N, M_tubular, 2, 3).reshape(N, M_tubular*2, 3)  # 形状 (N, M_tubular*2, 3)
        Pi_flat = Pi_valid.unsqueeze(-2).expand(N, M_tubular, 2, 3).reshape(N, M_tubular*2, 3)  # 形状 (N, M_tubular*2, 3)
        center = Pi_flat + t_flat.unsqueeze(-1) * Di_flat  # 形状 (N, M_tubular*2, 3)
        normal = torch.where(valid_flat.unsqueeze(-1), (center - intersect) / R, torch.zeros(N, M_tubular*2, 3, device=device))  # 形状 (N, M_tubular*2, 3)

        s_values.append(s_flat)  # 形状 (N, M_tubular*2)
        normals.append(normal)  # 形状 (N, M_tubular*2, 3)
        valid_flats.append(valid_flat)  # 形状 (N, M_tubular*2)

    # --- 端点球面交点（向量化） ---
    Pi = path_points.reshape(M_spherical, 3)  # 形状 (M_spherical, 3)

    # 边界框
    box_min = Pi - R  # 形状 (M_spherical, 3)
    box_max = Pi + R  # 形状 (M_spherical, 3)

    # 检查射线与边界框的 XY 交点
    mask_xy = (P[:, 0:1] >= box_min[:, 0]) & (P[:, 0:1] <= box_max[:, 0]) & \
              (P[:, 1:2] >= box_min[:, 1]) & (P[:, 1:2] <= box_max[:, 1])  # 形状 (N, M_spherical)

    if mask_xy.any():
        # 扩展维度
        Pi_exp = Pi.unsqueeze(0).expand(N, M_spherical, 3)  # 形状 (N, M_spherical, 3)
        P_exp = P[:, None, :].expand(N, M_spherical, 3)  # 形状 (N, M_spherical, 3)

        # 应用掩码
        valid_seg = mask_xy  # 形状 (N, M_spherical)
        inf_fill = torch.full_like(Pi_exp, float('inf'))  # 形状 (N, M_spherical, 3)
        Pi_valid = torch.where(valid_seg.unsqueeze(-1), Pi_exp, inf_fill)  # 形状 (N, M_spherical, 3)

        # 计算距离平方
        d2 = (P_exp[:, :, 0] - Pi_valid[:, :, 0])**2 + (P_exp[:, :, 1] - Pi_valid[:, :, 1])**2  # 形状 (N, M_spherical)
        d2 = torch.where(torch.isfinite(d2), d2, R**2 + 1)  # 无效 d2 设为不可交
        valid_d2 = (d2 < R**2) & valid_seg  # 形状 (N, M_spherical)
        #print(d2[0::101, :8][10]) 
        #print(valid_d2[0::101, :8][10]);exit()
        # 计算 s
        sqrt_term = torch.sqrt(torch.where(valid_d2, R**2 - d2, torch.zeros_like(d2)))  # 形状 (N, M_spherical)
        s1 = Pi_valid[:, :, 2] - P_exp[:, :, 2] + sqrt_term  # 形状 (N, M_spherical)
        s2 = Pi_valid[:, :, 2] - P_exp[:, :, 2] - sqrt_term  # 形状 (N, M_spherical)
        s = torch.stack([s1, s2], dim=-1)  # 形状 (N, M_spherical, 2)
        valid_s = (s >= 0) & valid_d2.unsqueeze(-1) & torch.isfinite(s)  # 形状 (N, M_spherical, 2)

        # 展平
        s_flat = s.view(N, M_spherical*2)  # 形状 (N, M_spherical*2)
        valid_flat = valid_s.view(N, M_spherical*2)  # 形状 (N, M_spherical*2)

        # 计算交点和法向量
        intersect = P[:, None, :] + s_flat.unsqueeze(-1) * torch.tensor([0, 0, 1], device=device)  # 形状 (N, M_spherical*2, 3)
        Pi_flat = Pi_valid.unsqueeze(-2).expand(N, M_spherical, 2, 3).reshape(N, M_spherical*2, 3)  # 形状 (N, M_spherical*2, 3)
        normal = torch.where(valid_flat.unsqueeze(-1), (Pi_flat - intersect) / R, torch.zeros(N, M_spherical*2, 3, device=device))  # 形状 (N, M_spherical*2, 3)

        s_values.append(s_flat)  # 形状 (N, M_spherical*2)
        #print(s_flat)
        #print(s_flat[valid_flat])
        normals.append(normal)  # 形状 (N, M_spherical*2, 3)
        valid_flats.append(valid_flat)  # 形状 (N, M_spherical*2)

    # --- 可微默认值 ---
    default_s = path_points[0, 0, 2] * 0 + MAX_HEIGHT  # 标量
    default_normal = torch.tensor([0.0, 0.0, 1.0 + path_points[0, 0, 2] * 0], device=device)  # 形状 (3,)
    default_s = default_s.expand(N)  # 形状 (N,)
    default_normal = default_normal.expand(N, 3)  # 形状 (N, 3)

    # --- 合并交点 ---
    if s_values:
        s_values = torch.cat(s_values, dim=1)  # 形状 (N, M_tubular*2 + M_spherical*2)
        normals = torch.cat(normals, dim=1)  # 形状 (N, M_tubular*2 + M_spherical*2, 3)
        valid_flat = torch.cat(valid_flats, dim=1)  # 形状 (N, M_tubular*2 + M_spherical*2)

        """ for i in range(N):            
            flag = False
            index = torch.argmin(s_values[i, :])  # 找到最小的 s 值的索引
            if torch.abs(normals[i, index, 0]) > 1e-6:
                flag = True
                print(P[i])
                for j in range(M_tubular*2 + M_spherical*2):   
                    if j < M_tubular*2:
                        print(f"Tube segment: i: {i}, j: {j}, s: {s_values[i, j]}, normal: {normals[i, j]}")
                    else:
                        
                        print(f"Spherical segment: i: {i}, j: {j}, s: {s_values[i, j]}, normal: {normals[i, j]}")
            if(flag==True):
                exit() """


        # 无效 s 设为 Z_INF，避免 0.0 * inf
        s_values_valid = torch.where(valid_flat, s_values, torch.ones_like(s_values) * Z_INF)  # 形状 (N, M_tubular*2 + M_spherical*2)
        s_values_valid = torch.cat([s_values_valid, default_s.unsqueeze(1)], dim=1)  # 形状 (N, M_tubular*2 + M_spherical*2 + 1)
        normals = torch.cat([normals, default_normal.unsqueeze(1)], dim=1)  # 形状 (N, M_tubular*2 + M_spherical*2 + 1, 3)
        
        # 在这里我们直接利用 argmin + gather 来获取最小值，以限制梯度反向传播的范围
        min_indices = torch.argmin(s_values_valid, dim=1)
        s_min = torch.gather(s_values_valid, 1, min_indices.unsqueeze(-1)).squeeze(-1)
        normal = torch.gather(normals, 1, min_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 3)).squeeze(1)

        height = P[:, 2] + s_min  # 形状 (N,)
        normal = normal / (torch.norm(normal, dim=-1, keepdim=True))  # 形状 (N, 3)
    else:
        height = default_s  # 形状 (N,)
        normal = default_normal  # 形状 (N, 3)

    #print(f"Height: {height}, Normal: {normal}");exit()
    return height, normal

# 示例使用
if __name__ == "__main__":
    P = torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.1, 0.0]])  # 形状 (N=2, 3)
    R = 0.1
    path_points = torch.tensor([
        [[0.0, 0.0, 0.1], [0.1, 0.0, 0.1], [0.2, 0.1, 0.1]],
        [[0.0, 0.2, 0.3], [0.1, 0.2, 0.3], [0.2, 0.3, 0.3]]
    ])  # 形状 (path_num=2, j-i+1=3, 3)
    
    height, normal = ray_sweep_intersection(P, R, path_points, device='cpu')
    print(f"Height: {height.tolist()}")
    print(f"Normal: {normal.tolist()}")