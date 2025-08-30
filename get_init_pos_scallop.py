import torch
import config as cfg
from old_code.NURBS import NURBS
import resample
import numpy as np
import utils

def calculate_stepover_distance_batch(R_tool: float, h: float, k1: torch.Tensor, k2: torch.Tensor) -> torch.Tensor:
    """
    根据一批局部曲面曲率，批量计算保持恒定残余高度(h)所需的刀具行距(d)。

    该函数使用PyTorch张量操作，高效地处理整行数据。

    参数:
        R_tool (float):         球头刀的半径。
        h (float):              目标残余高度。
        k1 (torch.Tensor):      形状为[N,]的第一个主曲率张量。
        k2 (torch.Tensor):      形状为[N,]的第二个主曲率张量。

    曲率符号约定:
        - k > 0: 凹面
        - k < 0: 凸面
        - k = 0: 平面

    返回:
        torch.Tensor: 形状为[N,]的计算出的安全行距d。发生干涉的位置返回0.0。
    """
    # --- 输入验证 ---
    if R_tool <= 0 or h <= 0:
        raise ValueError("刀具半径和残余高度必须是正数。")
    if k1.shape != k2.shape:
        raise ValueError("k1 和 k2 张量的形状必须相同。")

    # --- 步骤 1: 确定每个点的最严格曲率 ---
    k_surface = torch.maximum(k1, k2)

    # 初始化等效半径张量 R_eff，默认值为平面情况下的刀具半径
    R_eff = torch.full_like(k_surface, fill_value=R_tool)
    
    # 用于浮点数比较的极小值
    epsilon = 1e-9

    # --- 步骤 2: 使用布尔掩码识别凹面和凸面 ---
    is_convex = k_surface < -epsilon
    is_concave = k_surface > epsilon

    # --- 步骤 3: 批量处理凸面情况 ---
    if torch.any(is_convex):
        # 只对凸面部分计算曲率半径
        R_surface_convex = torch.abs(1.0 / k_surface[is_convex])
        # 计算凸面部分的等效半径
        R_eff_convex = (R_tool * R_surface_convex) / (R_surface_convex - R_tool)
        # 将计算结果更新到 R_eff 张量的对应位置
        R_eff[is_convex] = R_eff_convex

    # --- 步骤 4: 批量处理凹面情况 (包含干涉检查) ---
    if torch.any(is_concave):
        # 只对凹面部分计算曲率半径
        R_surface_concave = 1.0 / k_surface[is_concave]
        
        # 关键: 批量检查干涉
        interferes = R_tool >= R_surface_concave
        if torch.any(interferes):
            # 找出实际发生干涉的点在原始张量中的索引
            concave_indices = torch.where(is_concave)[0]
            interfering_indices = concave_indices[interferes]
            print(f"警告: 在 {len(interfering_indices)} 个点上发生干涉！"
                  f"刀具半径({R_tool}) >= 凹面曲率半径。这些点的行距将设为0。")
        
        # 计算未发生干涉的凹面部分的等效半径
        # ~interferes 是对布尔张量取反，选出"非干涉"的点
        valid_concave_mask = ~interferes
        if torch.any(valid_concave_mask):
            R_eff_concave = (R_tool * R_surface_concave[valid_concave_mask]) / \
                            (R_surface_concave[valid_concave_mask] + R_tool)
            
            # 更新 R_eff 张量中 "有效的凹面" 部分
            # is_concave.clone() 创建一个副本以安全地修改
            valid_concave_global_mask = is_concave.clone()
            valid_concave_global_mask[is_concave] = valid_concave_mask
            R_eff[valid_concave_global_mask] = R_eff_concave
        
        # 对于发生干涉的凹面，它们的 R_eff 会保持默认值 R_tool，
        # 但我们将在最后一步通过检查 (2*R_eff-h) 来确保它们的行距为0。

    # --- 步骤 5: 批量计算最终行距 d ---
    # 使用统一公式 d = 2 * sqrt(h * (2 * R_eff - h))
    inner_value = h * (2 * R_eff - h)

    # 创建一个安全掩码，将所有 inner_value < 0 的位置标记为False
    # 这会自动处理干涉情况（因为 R_tool > R_surface 会导致 R_eff 为负）
    safe_mask = inner_value >= 0
    
    # 初始化输出张量为0
    d_out = torch.zeros_like(k_surface)
    
    # 只对安全的位置计算行距
    d_out[safe_mask] = 2 * torch.sqrt(inner_value[safe_mask])
    
    return d_out

def get_next_points(h, sample_points, bsurface:NURBS, control_points, wij, device='cpu'):
    """
    计算下一行采样点的位置
    
    输入：
        h: 残余高度，标量值。
        sample_points: 形状为 [N, 2] 的张量，表示采样点的坐标。
        bsurface: NURBS 曲面对象。
        control_points: 控制点张量。
        wij: 权重张量。
        device: 设备类型，默认为 'cpu'。
        
    输出：
        下一行采样点的位置。
    """
    # 获取曲面上的法向量和主曲率
    heights, normals, k1, k2, Jacobians = bsurface.evaluate_curvature_batch(sample_points, control_points, wij, batch_size=500000)
    positions = torch.cat((sample_points, heights.unsqueeze(-1)), dim=-1)  # [N, 3]
    # 前进方向
    forward_dirs = positions[2:, :] - positions[:-2, :]  # [N-2, 3]
    forward_dirs = torch.cat((positions[1:2, :]-positions[0:1, :], forward_dirs, positions[-1:, :]-positions[-2:-1, :]), dim=0)  # [N, 3]
    next_dirs = torch.cross(normals, forward_dirs, dim=-1)  # [N, 3]
    next_dirs = next_dirs / torch.norm(next_dirs, dim=-1, keepdim=True)  # 归一化
    if(torch.isnan(next_dirs).any()):
        print("Warning: next_dirs contains NaN values, replacing with zeros.")
        print(normals[torch.isnan(next_dirs)], forward_dirs[torch.isnan(next_dirs)], positions[torch.isnan(next_dirs)]);exit()
        next_dirs = torch.zeros_like(next_dirs)
    
    r = cfg.R  # 刀具半径
    d = calculate_stepover_distance_batch(r, h, k1, k2)  # [N,]
    
    if(torch.isnan(d).any()):
        print("Warning: d contains NaN values, replacing with zeros.")
        d = torch.zeros_like(d)
    
    next_points = positions + next_dirs * d.unsqueeze(-1)  # [N, 3]
    now_centers = positions + normals * r  # [N, 3]
    #delta_P = next_dirs * d.unsqueeze(-1)  # [N, 3]

    return next_points, now_centers, Jacobians, positions

def project_to_next_row_uv_batch(
    current_uvs: torch.Tensor,
    current_points: torch.Tensor,
    next_points_3d_candidates: torch.Tensor,
    jacobians: torch.Tensor
) -> "tuple[torch.Tensor, torch.Tensor]":
    """
    使用线性近似法，批量计算下一行刀具路径的UV坐标。

    此函数将3D空间中的候选点(candidates)投影回UV参数空间。

    参数:
        current_uvs (torch.Tensor):      当前行路径点的UV坐标，形状为 [N, 2]。
        current_points (torch.Tensor):   当前行路径点的3D坐标，形状为 [N, 3]。
        next_points_3d_candidates (torch.Tensor): 下一行路径的3D候选点坐标
                                                  (由 P_i + d * V_step 计算得到)，形状为 [N, 3]。
        jacobians (torch.Tensor):        在每个current_point处的雅可比矩阵，形状为 [N, 3, 2]。

    返回:
        tuple[torch.Tensor, torch.Tensor]:
        - next_uvs (torch.Tensor): 计算出的下一行路径点的UV坐标，形状为 [N, 2]。
        - valid_mask (torch.Tensor): 一个布尔掩码，形状为 [N,]。
                                     如果对应的next_uv在[0,1]范围内，则为True，否则为False。
    """
    # --- 步骤 1: 计算期望的3D位移向量 delta_P ---
    # delta_P 的形状为 [N, 3]
    delta_P = next_points_3d_candidates - current_points
    # 为矩阵乘法调整形状为 [N, 3, 1]
    delta_P = delta_P.unsqueeze(-1)

    # --- 步骤 2: 批量计算雅可比矩阵的伪逆 ---
    # torch.linalg.pinv 可以直接处理批量数据
    # 输入 jacobians [N, 3, 2]，输出 Js_pinv [N, 2, 3]
    try:
        Js_pinv = utils.robust_pinv(jacobians, alpha=1e-4)  # 使用 utils 中的 robust_pinv 函数来计算伪逆
        #Js_pinv = torch.linalg.pinv(jacobians)
    except torch.linalg.LinAlgError as e:
        print(f"错误: 伪逆计算失败，可能因为雅可比矩阵包含奇异矩阵。 {e}")
        # 在失败时返回空结果和全False的掩码
        return torch.zeros_like(current_uvs), torch.zeros(current_uvs.shape[0], dtype=torch.bool)

    # --- 步骤 3: 批量求解参数空间的位移 delta_uv ---
    # Js_pinv [N, 2, 3] @ delta_P [N, 3, 1] -> delta_uv [N, 2, 1]
    #print(Js_pinv[:10], delta_P[:10])
    delta_uv = Js_pinv.to(delta_P.device) @ delta_P
    # 将形状从 [N, 2, 1] 压缩回 [N, 2]
    delta_uv = delta_uv.squeeze(-1)

    # --- 步骤 4: 计算下一行的UV坐标 ---
    next_uvs = current_uvs + delta_uv

    # --- 步骤 5: 边界检查并生成有效性掩码 ---
    # 检查u坐标是否在[0, 1]范围内
    u_valid = (next_uvs[:, 0] >= 0.0) & (next_uvs[:, 0] <= 1.0)
    # 检查v坐标是否在[0, 1]范围内
    v_valid = (next_uvs[:, 1] >= 0.0) & (next_uvs[:, 1] <= 1.0)
    
    # 最终的掩码要求u和v都有效
    valid_mask = u_valid & v_valid
    
    return next_uvs, valid_mask

def trivial_project_to_next_row_uv(current_uvs: torch.Tensor,
    current_points: torch.Tensor,
    next_points_3d_candidates: torch.Tensor,
    jacobians: torch.Tensor
) -> "tuple[torch.Tensor, torch.Tensor]":
    next_uvs = next_points_3d_candidates[:, :2]  # 直接取前两列作为UV坐标
    # 检查UV坐标是否在[0, 1]范围内
    u_valid = (next_uvs[:, 0] >= 0.0) & (next_uvs[:, 0] <= 1.0)
    v_valid = (next_uvs[:, 1] >= 0.0) & (next_uvs[:, 1] <= 1.0)
    valid_mask = u_valid & v_valid  # 最终的掩码要求u和v都有效
    return next_uvs, valid_mask

def calculate_boundary_intersection_batch(
    prev_points_uv: torch.Tensor,
    next_points_uv: torch.Tensor
) -> torch.Tensor:
    """
    批量计算从有效点(prev)到无效点(next)的线段与UV域[0,1]x[0,1]的交点。

    此函数完全向量化，没有Python循环，计算效率高。

    参数:
        prev_points_uv (torch.Tensor):  有效的起始点UV坐标，形状 [N, 2]。
        next_points_uv (torch.Tensor):  无效的终点UV坐标，形状 [N, 2]。

    返回:
        torch.Tensor: 精确的边界交点UV坐标，形状 [N, 2]。
    """
    # --- 步骤 1: 准备计算 ---
    # 方向向量 delta = P_next - P_prev
    delta = next_points_uv - prev_points_uv
    
    # 获取设备信息，并设置一个极小值epsilon以避免除以零
    device = prev_points_uv.device
    epsilon = 1e-14

    # --- 步骤 2: 并行计算与所有4个边界相交的插值参数 t ---
    # 初始化t值为一个很大的数（代表不相交）
    t = torch.full((prev_points_uv.shape[0], 4), float('inf'), device=device)

    # 对于 u=0 边界: t = (0 - u_prev) / delta_u
    # 只在delta.x != 0时计算
    mask_u0 = torch.abs(delta[:, 0]) > epsilon
    t[mask_u0, 0] = (0.0 - prev_points_uv[mask_u0, 0]) / delta[mask_u0, 0]

    # 对于 u=1 边界: t = (1 - u_prev) / delta_u
    mask_u1 = torch.abs(delta[:, 0]) > epsilon
    t[mask_u1, 1] = (1.0 - prev_points_uv[mask_u1, 0]) / delta[mask_u1, 0]

    # 对于 v=0 边界: t = (0 - v_prev) / delta_v
    mask_v0 = torch.abs(delta[:, 1]) > epsilon
    t[mask_v0, 2] = (0.0 - prev_points_uv[mask_v0, 1]) / delta[mask_v0, 1]

    # 对于 v=1 边界: t = (1 - v_prev) / delta_v
    mask_v1 = torch.abs(delta[:, 1]) > epsilon
    t[mask_v1, 3] = (1.0 - prev_points_uv[mask_v1, 1]) / delta[mask_v1, 1]

    # --- 步骤 3: 屏蔽无效的 t 值 ---
    # 一个有效的交点必须在原线段内部，即 0 < t <= 1
    # 我们允许 t=0 的情况以处理起始点就在边界上的情况
    # 将所有超出范围 [0, 1] 的 t 值设为无穷大，使其在求最小值时被忽略
    t[ (t < 0.0) | (t > 1.0) ] = float('inf')

    # --- 步骤 4: 找到第一个交点 (最小的有效 t 值) ---
    # t_final 的形状为 [N,]，包含了每条线段对应的最小t值
    t_final, _ = torch.min(t, dim=1)

    # --- 步骤 5: 处理没有有效交点的情况 ---
    # 如果一条线段的所有t值都是inf（例如，平行于边界且在界外），
    # t_final会是inf。这在理论上不应发生，因为P_prev在界内而P_next在界外。
    # 但作为安全措施，我们将这些情况的交点设为起始点。
    no_intersection_mask = torch.isinf(t_final)
    if torch.any(no_intersection_mask):
        print(prev_points_uv, next_points_uv);exit()
        print(f"警告: 在 {no_intersection_mask.sum()} 个点上未找到有效的边界交点。将使用起始点作为替代。")
        t_final[no_intersection_mask] = 0.0 # 使用起始点

    # --- 步骤 6: 使用最终的 t 值进行线性插值 ---
    # P_intersect = P_prev + t_final * delta
    # t_final需要扩展维度以进行广播: [N,] -> [N, 1]
    intersection_points = prev_points_uv + t_final.unsqueeze(1) * delta

    return intersection_points

def build_segmented_path(
    path_uvs: torch.Tensor,
    valid_mask: torch.Tensor
) -> "list[torch.Tensor]":
    """
    接收一行包含无效点的UV坐标，返回一个由精确边界点分割的、不连续的路径段列表。
    
    参数:
        path_uvs (torch.Tensor): 包含出界点的原始UV路径，形状 [N, 2]。
        valid_mask (torch.Tensor): 对应的有效性布尔掩码，形状 [N,]。

    返回:
        list[torch.Tensor]: 一个列表，其中每个元素都是一个Tensor，代表一段连续的有效路径。
                            例如：[[P_start, P1, P2, P_exit], [P_enter, P3, P_exit_2]]
    """
    # 如果路径中没有点，或所有点都无效，则返回空列表
    if path_uvs.shape[0] == 0 or not torch.any(valid_mask):
        return []

    # --- 步骤 1: 向量化检测状态转换点 ---
    if path_uvs.shape[0] > 1:
        current_valid = valid_mask[:-1]
        next_valid = valid_mask[1:]
        is_exit_point_mask = current_valid & ~next_valid
        is_entry_point_mask = ~current_valid & next_valid
    else: # 只有一个点的情况
        is_exit_point_mask = torch.tensor([False], device=path_uvs.device)
        is_entry_point_mask = torch.tensor([False], device=path_uvs.device)

    # --- 步骤 2: 批量计算所有边界交点 ---
    # 创建一个原始路径的副本，我们将用精确交点来更新它
    refined_path_uvs = path_uvs.clone()

    # 处理所有“出界”点
    if torch.any(is_exit_point_mask):
        # 找出所有出界线段的起点 (P_prev) 和终点 (P_next)
        exit_prev_uvs = path_uvs[:-1][is_exit_point_mask]
        exit_next_uvs = path_uvs[1:][is_exit_point_mask]
        #print(exit_prev_uvs, exit_next_uvs)
        # 批量计算交点
        exit_intersections = calculate_boundary_intersection_batch(exit_prev_uvs, exit_next_uvs)
        # 将计算出的交点更新回 refined_path 的对应位置
        # 注意：这里更新的是出界线段的 *终点* 位置 (即那个无效点的位置)
        refined_path_uvs[1:][is_exit_point_mask] = exit_intersections

    # 处理所有“入界”点
    if torch.any(is_entry_point_mask):
        # 找出所有入界线段的起点 (P_prev) 和终点 (P_next)
        entry_prev_uvs = path_uvs[:-1][is_entry_point_mask]
        entry_next_uvs = path_uvs[1:][is_entry_point_mask]
        #print(entry_prev_uvs, entry_next_uvs)
        #print(is_entry_point_mask)
        # 批量计算交点
        entry_intersections = calculate_boundary_intersection_batch(entry_next_uvs, entry_prev_uvs)
        # 将计算出的交点更新回 refined_path 的对应位置
        # 注意：这里更新的是入界线段的 *起点* 位置 (即那个无效点的位置)
        refined_path_uvs[:-1][is_entry_point_mask] = entry_intersections


    # --- 步骤 3: 处理路径起点和终点的特殊情况 ---
    # 如果第一个点就无效，它需要一个“入界”交点
    if not valid_mask[0] and path_uvs.shape[0] > 1:
        intersection = calculate_boundary_intersection_batch(path_uvs[1].unsqueeze(0), path_uvs[0].unsqueeze(0))
        refined_path_uvs[0] = intersection.squeeze(0)
        
    # 如果最后一个点无效，它需要一个“出界”交点
    if not valid_mask[-1] and path_uvs.shape[0] > 1:
        intersection = calculate_boundary_intersection_batch(path_uvs[-2].unsqueeze(0), path_uvs[-1].unsqueeze(0))
        refined_path_uvs[-1] = intersection.squeeze(0)

    # --- 步骤 4: 根据原始的 valid_mask 分割路径 ---
    segmented_paths = []
    start_idx = -1
    for i in range(len(valid_mask)):
        if valid_mask[i] and start_idx == -1:
            # 找到一个新路径段的开始
            start_idx = i
        elif not valid_mask[i] and start_idx != -1:
            # 找到一个路径段的结束
            # 该段的范围是 [start_idx, i]
            # 注意我们取到 i，因为 refined_path[i] 已经被更新为精确的出界交点了
            segmented_paths.append(refined_path_uvs[start_idx : i + 1])
            start_idx = -1
        elif i<len(valid_mask)-1 and not valid_mask[i] and valid_mask[i+1]:
            start_idx = i
            # 更新后的入点
    
    # 处理循环结束后仍未结束的最后一个路径段
    if start_idx != -1:
        segmented_paths.append(refined_path_uvs[start_idx:])

    return segmented_paths

def make_it_continuous(segmented_paths):
    for i, segment in enumerate(segmented_paths):
        if i==0:
            res = segment
        else:
            res = torch.cat([res, segment])
    return res
    

def get_nurbs_data(device='cpu'):
    old_project_path = '/data/wzr/2025'
    control_points = torch.load(f'{old_project_path}/control_points1.pth').to(device)
    wij = torch.load(f'{old_project_path}/wij1.pth').to(device)
    #control_points = torch.ones((100, 100)).to(device)
    #wij = torch.ones((100, 100)).to(device)

    bsurface = NURBS(control_points, degree_u=3, degree_v=3, sample_points=None)
    
    return bsurface, control_points, wij

def generate_sample_points(x_num=cfg.path_size+1, y_num=cfg.path_num, device='cpu'):
    """
    生成第一组点，形状 [x_num, 3]
    """
    xs = torch.linspace(0., 1., x_num).to(device)  # x 方向的采样点
    sample_points = torch.cat([xs[:, None], torch.zeros((x_num, 1), device=device)], dim=-1)  # [x_num, 2]
    #print(sample_points.shape)
    return sample_points

def reshape_data(centers, x_num=cfg.path_size + 1, y_num=cfg.path_num):
    """
    上面得到的 surface_points 是 [N, 3] 的张量，并按先变 y 再变 x 的顺序排列
    现在要将其 reshape 成 [path_num, path_size + 1, 3] 的张量，
    """
    centers = centers.reshape(x_num, y_num, 3)
    centers = centers.transpose(0, 1)  # 转置为 [path_num, path_size + 1, 3]
    return centers


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
    centers=[]
    h = cfg.h
    bsurface, control_points, wij = get_nurbs_data(device)
    print("开始生成路径。")
    while(True):   
        next_positions, now_centers, Jacobians, positions = get_next_points(h, sample_points, bsurface, control_points, wij, device=device)
        #print(next_positions)
        centers.append(now_centers)
        #next_uvs, valid_mask = project_to_next_row_uv_batch(sample_points, positions, next_positions, Jacobians)
        next_uvs, valid_mask = trivial_project_to_next_row_uv(sample_points, positions, next_positions, Jacobians)
        segmented = build_segmented_path(next_uvs, valid_mask)
        if len(segmented) == 0:
            print("没有有效的路径段，停止迭代。")
            break
        sample_points = make_it_continuous(segmented)
    print(f"初始化路径为{len(centers)}条.")
    cfg.path_num = len(centers)
    return centers
    
def get_resampled_init_pos_tensor(device='cpu'):
    init_pos_list = get_init_pos(device)
    new_centers = np.zeros((cfg.path_num, cfg.path_size+1, 3))  # 转换为 NumPy 数组以便使用 resample 函数
    for idx in range(len(init_pos_list)):
        new_centers[idx] = resample.resample_toolpath_by_x(init_pos_list[idx].cpu().numpy(), cfg.path_size+1)
    new_centers = torch.tensor(new_centers, device=device, dtype=torch.float32)  # 转换回张量并返回
    #print(new_centers)
    #print(new_centers.shape)
    return new_centers

if __name__ == "__main__":
    # --- 示例 ---
    init_pos_list = get_init_pos(device='cuda:0')
    new_centers = get_resampled_init_pos_tensor(init_pos_list, device='cuda:0')