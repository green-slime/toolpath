import torch

def compute_laplacian_smoothness_loss(path_points, laplacian_weight=0.01):
    """
    拉普拉斯正则项：每个点与左右两点平均位置的L2距离
    
    Args:
        path_points: [path_num, path_len, 3] 路径点
        laplacian_weight: 拉普拉斯正则项权重
    
    Returns:
        拉普拉斯平滑损失
    """
    path_num, path_len, _ = path_points.shape
    
    if path_len < 3:
        return torch.tensor(0.0, device=path_points.device, requires_grad=True)
    
    # 获取内部点（排除首尾点）
    center_points = path_points[:, 1:-1, :]  # [path_num, path_len-2, 3]
    
    # 获取左右邻居点
    left_points = path_points[:, :-2, :]     # [path_num, path_len-2, 3]
    right_points = path_points[:, 2:, :]     # [path_num, path_len-2, 3]
    
    # 计算左右邻居的平均位置
    neighbor_average = (left_points + right_points) / 2.0  # [path_num, path_len-2, 3]
    
    # 计算每个内部点与其邻居平均位置的L2距离
    laplacian_residuals = center_points - neighbor_average  # [path_num, path_len-2, 3]
    
    # L2 损失
    laplacian_loss = torch.sum(laplacian_residuals ** 2)
    
    return laplacian_weight * laplacian_loss


def compute_angle_smoothness_loss_vectorized(path_points, angle_weight=0.08):
    """
    向量化版本：约束路径转向角度变化的平滑性
    path_points: [path_num, path_len, 3]
    """
    path_num, path_len, _ = path_points.shape
    
    if path_len < 3:
        return torch.tensor(0.0, device=path_points.device, requires_grad=True)
    
    # 计算所有路径的相邻段向量
    # seg1: [path_num, path_len-2, 3] - 第一段向量
    seg1 = path_points[:, 1:-1, :] - path_points[:, :-2, :]
    # seg2: [path_num, path_len-2, 3] - 第二段向量  
    seg2 = path_points[:, 2:, :] - path_points[:, 1:-1, :]
    
    # 向量化归一化
    seg1_norm = seg1 / (torch.norm(seg1, dim=2, keepdim=True) + 1e-8)  # [path_num, path_len-2, 3]
    seg2_norm = seg2 / (torch.norm(seg2, dim=2, keepdim=True) + 1e-8)  # [path_num, path_len-2, 3]
    
    # 向量化计算点积（余弦值）
    cos_angles = torch.sum(seg1_norm * seg2_norm, dim=2)  # [path_num, path_len-2]
    
    # 计算角度惩罚
    angle_penalty = torch.sum((1 - cos_angles) ** 2)  # 标量

    return angle_weight * angle_penalty


def compute_curvature_loss(path_points, curvature_weight=0.1):
    """
    推荐的稳定曲率计算
    """
    path_num, path_len, _ = path_points.shape
    
    if path_len < 3:
        return torch.tensor(0.0, device=path_points.device, requires_grad=True)
    
    # 连续三点
    p0 = path_points[:, :-2, :]   # [path_num, path_len-2, 3]
    p1 = path_points[:, 1:-1, :]  # [path_num, path_len-2, 3]
    p2 = path_points[:, 2:, :]    # [path_num, path_len-2, 3]
    
    # 边向量
    v1 = p1 - p0
    v2 = p2 - p1

    epsilon = 1e-8
    
    # 边长
    len1 = torch.norm(v1, dim=2) + epsilon
    len2 = torch.norm(v2, dim=2) + epsilon
    
    # 叉积模长（面积）
    cross_product = torch.cross(v1, v2, dim=2)
    area = torch.norm(cross_product, dim=2)
    
    # 第三边长度
    v3 = p2 - p0
    len3 = torch.norm(v3, dim=2) + epsilon
    
    # 外接圆半径的倒数（曲率）
    # area 可以为 0
    curvature = 4 * area / (len1 * len2 * len3)
    
    return curvature_weight * torch.sum(curvature ** 2)