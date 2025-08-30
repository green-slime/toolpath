import torch
import torch.nn.functional as F

def compute_refraction_direction(normal, incident_dir, n1=1.5, n2=1.0):
    """
    计算折射光线方向
    参数:
        normal: 表面法向量 [N, 3]
        incident_dir: 入射光方向 [N, 3] 或 [3]
        n1: 入射介质折射率（玻璃）
        n2: 折射介质折射率（空气）
    返回:
        refracted_dir: 折射光方向 [N, 3]
    """
    # 确保法向量和入射方向都是规范化的
    normal = F.normalize(normal, p=2, dim=1)
    with torch.no_grad():
        if len(incident_dir.shape) == 1:
            incident_dir = incident_dir.unsqueeze(0).expand(normal.shape[0], -1)
        incident_dir = F.normalize(incident_dir, p=2, dim=1)
    
    # 计算入射角的余弦值
    cos_i = torch.sum(incident_dir * normal, dim=1, keepdim=True)
    
    # 使用斯涅尔定律
    n = n1 / n2
    cos_t_sq = 1.0 - n * n * (1.0 - cos_i * cos_i)
    
    # 检查是否发生全反射
    valid_mask = cos_t_sq >= 0
    cos_t = torch.sqrt(torch.clamp(cos_t_sq, min=0.0))
    
    # 计算折射方向
    refracted_dir = n * incident_dir - (n * cos_i - cos_t) * normal
    refracted_dir = F.normalize(refracted_dir, p=2, dim=1)
    
    # 对于全反射的情况，返回反射方向
    reflected_dir = incident_dir + 2 * cos_i * normal
    refracted_dir = torch.where(valid_mask, refracted_dir, reflected_dir)
    
    return refracted_dir

def compute_intersection_with_plane(origins, directions, z_plane):
    """
    计算光线与平面的交点
    参数:
        origins: 光线起点 [N, 3]
        directions: 光线方向 [N, 3]
        z_plane: 接收平面的z坐标（标量）
    返回:
        intersection_points: 交点坐标 [N, 2]（只返回x,y坐标）
    """
    # 计算t参数：(z_plane - z0) / dz
    t = (z_plane - origins[:, 2]) / directions[:, 2]
    
    # 计算交点
    intersection_points = origins + t.unsqueeze(1) * directions
    
    # 只返回x,y坐标
    return intersection_points[:, :2]

def trace_rays_through_surface(surface_points, surface_normals, z_receiver, n1=1.5, n2=1.0):
    """
    追踪光线从曲面出射到接收平面
    参数:
        surface_points: 曲面上的点 [N, 3]
        surface_normals: 曲面法向量 [N, 3]
        z_receiver: 接收平面的z坐标
        n1: 玻璃折射率
        n2: 空气折射率
    返回:
        receiver_points: 接收平面上的点 [N, 2]
    """
    # 入射光方向（从下往上）
    incident_dir = torch.tensor([0.0, 0.0, 1.0], device=surface_points.device, requires_grad=False)
    
    # 计算折射方向
    refracted_dirs = compute_refraction_direction(surface_normals, incident_dir, n1, n2)
    
    # 计算与接收平面的交点
    receiver_points = compute_intersection_with_plane(
        surface_points, 
        refracted_dirs, 
        z_receiver
    )
    
    return receiver_points