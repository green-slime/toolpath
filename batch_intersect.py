import torch
from tqdm import tqdm
import bvh

Z_INF = 1e12  # 用于表示无效的高度

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
        path_points = bvh.batch_make_path_points(path_yzs, x_grid, k_batch)  # 获取可能相交的包络面
        height, normal= ray_sweep_intersection(Ps_k, R, path_points, device=device, alpha=alpha, MAX_HEIGHT=MAX_HEIGHT)
        heights.append(height)
        normals.append(normal)
    # test: backward()
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
        s1 = torch.where(valid, (-b + sqrt_d) / (2 * a + 1e-12), Z_INF)  # 形状 (N, M_tubular)
        s2 = torch.where(valid, (-b - sqrt_d) / (2 * a + 1e-12), Z_INF)  # 形状 (N, M_tubular)

        # 合并 s1 和 s2
        s = torch.stack([s1, s2], dim=-1)  # 形状 (N, M_tubular, 2)
        valid_s = (s >= 0) & valid.unsqueeze(-1)  # 形状 (N, M_tubular, 2)

        # 计算 t
        t = (torch.sum(Qi * Di, dim=-1, keepdim=True) + s * Di[:, :, 2:3]) / (Di_norm2.unsqueeze(-1) + 1e-12)  # 形状 (N, M_tubular, 2)
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

        # w = exp(-sqrt(s_values_valid - s_values_valid.min()) * alpha)
        #min_vals, _ = torch.min(s_values_valid, dim=1, keepdim=True)
        #weights = torch.softmax(-torch.sqrt((s_values_valid-min_vals)) * alpha, dim=1)  # 形状 (N, M_tubular*2 + M_spherical*2 + 1)
        weights = torch.softmax(-s_values_valid * alpha, dim=1)  # 形状 (N, M_tubular*2 + M_spherical*2 + 1)
        
        s_min = torch.sum(weights * s_values_valid, dim=1)  # 形状 (N,)
        """ for i in range(len(s_min)):
            if s_min[i] > 0.3003:
                print(weights[i], weights[i].shape)
                print(s_values_valid[i]);exit() """
            
        normal = torch.sum(weights.unsqueeze(-1) * normals, dim=1)  # 形状 (N, 3)
        height = P[:, 2] + s_min  # 形状 (N,)
        normal = normal / (torch.norm(normal, dim=-1, keepdim=True) + 1e-12)  # 形状 (N, 3)
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