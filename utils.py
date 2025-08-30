import torch

def cuda_init(gpu_index:int)->torch.device:
    """
    gpu_index: 用几号gpu
    """
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            device = torch.device(f"cuda:{gpu_index}")
            torch.cuda.set_device(device)
        else:
            device = torch.device('cpu')
            print("CUDA 不可用，使用 CPU")
        #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return device
    except Exception as e:
        raise RuntimeError(f'cuda 初始化出现问题：{str(e)}') from e
    
def make_1d_sample_points(grid_size=100, device='cpu'):
    """
    生成一维采样点
    返回一个形状为 (N, 1) 的张量。
    """
    x = torch.linspace(0., 1., grid_size + 1, device=device)
    sample_points = x.unsqueeze(-1)  # 将 x 转换为列向量
    return sample_points

def make_2d_sample_points(grid_size=100, device='cpu'):
    """
    生成二维采样点
    返回一个形状为 (N**2, 2) 的张量。
    """
    x = torch.linspace(0., 1., grid_size + 1, device=device)
    y = torch.linspace(0., 1., grid_size + 1, device=device)
    xx, yy = torch.meshgrid(x, y, indexing='ij')  # 使用 ij 索引方式
    sample_points = torch.stack([xx.flatten(), yy.flatten()], dim=-1)  # 将 x 和 y 合并
    return sample_points

def robust_pinv(J, alpha=1e-6):
    """
    计算雅可比矩阵J的正则化伪逆，以处理奇异矩阵。
    J 的形状为 [N, 3, 2]
    """
    # 批量计算 J.T @ J, 结果形状为 [N, 2, 2]
    JtJ = J.transpose(-2, -1) @ J
    
    # 创建一个与JtJ形状匹配的单位矩阵
    # eye_matrix 的形状为 [2, 2]
    eye_matrix = torch.eye(J.shape[2], device=J.device)
    
    # 给对角线加上一个微小的扰动值 alpha
    # (JtJ + alpha * I)
    JtJ_regularized = JtJ + alpha * eye_matrix
    
    # 求解这个正则化后的线性系统
    # (JtJ + alpha * I)^-1 * J.T
    try:
        J_pinv_robust = torch.linalg.solve(JtJ_regularized, J.transpose(-2, -1))
        return J_pinv_robust
    except torch.linalg.LinAlgError:
        # 如果加上扰动后仍然奇异（极罕见），则返回一个零矩阵作为最终回退
        print("警告: 正则化后仍然遇到奇异矩阵，返回零伪逆。")
        return torch.zeros_like(J.transpose(-2, -1))
    
import os

def create_closed_grid(grid_size, heights, normals, min_height=None, output_file="output/output.obj"):
    """
    生成封闭网格（包括顶面、底面和侧面），保存为 OBJ 文件
    参数：
        grid_size: grid_coords 用 make_2d_sample_points(grid_size) 函数生成，以保持和 coords 一致
        heights: [N²,]，每个网格点的高度 (z)
        normals: [N², 3]，顶面顶点的法向量 (nx, ny, nz)
        min_height: 若设置为 None，则底面高度为 heights 的最小值减去 0.01，否则为指定的最小高度
        output_file: 输出 OBJ 文件路径
    返回：
        vertices: [2*N², 3]，所有顶点（顶面 + 底面）
        faces: [F, 3]，三角形面索引
        vertex_normals: [2*N², 3]，所有顶点法向量
    """
    print("Creating closed grid...")
    try:
        if not isinstance(heights, torch.Tensor):
            heights = torch.tensor(heights, dtype=torch.float32)
        if not isinstance(normals, torch.Tensor):
            normals = torch.tensor(normals, dtype=torch.float32)
        if isinstance(min_height, (int, float)):
            min_height = torch.tensor(min_height, dtype=torch.float32)

        os.makedirs(os.path.dirname(output_file), exist_ok=True)  # 确保目录存在
        
        N = grid_size + 1  # N 个点

        grid_coords = make_2d_sample_points(grid_size, device=heights.device)  # [N², 2]

        # 底面高度为高度数组的最小值
        if min_height == None:        
            min_height = heights.min()-0.01

        # 顶面顶点：(x, y, z)
        top_vertices = torch.cat([grid_coords, heights.unsqueeze(-1)], dim=-1)  # [N², 3]
        
        # 底面顶点：(x, y, min_height)
        bottom_vertices = torch.cat([grid_coords, torch.full_like(heights, min_height).unsqueeze(-1)], dim=-1)  # [N², 3]
        
        # 合并顶点：顶面 + 底面
        vertices = torch.cat([top_vertices, bottom_vertices], dim=0)  # [2*N², 3]

        # 法向量
        top_normals = normals  # 顶面法向量 [N², 3]
        bottom_normals = torch.zeros_like(top_normals)  # 底面法向量 [N², 3]
        bottom_normals[:, 2] = -1.0  # 底面朝下 (0, 0, -1)
        vertex_normals = torch.cat([top_normals, bottom_normals], dim=0)  # [2*N², 3]

        # 生成面
        faces = []
        face_normals = []  # 每个面的法向量索引

        # 顶面三角形
        for i in range(N - 1):
            for j in range(N - 1):
                v0 = i * N + j
                v1 = v0 + 1
                v2 = (i + 1) * N + j
                v3 = v2 + 1
                faces.append([v0, v1, v2])
                face_normals.append([v0, v2, v1])
                faces.append([v1, v3, v2])
                face_normals.append([v1, v2, v3])

        # 底面三角形（索引偏移 N²，翻转顺序以确保朝外）
        for i in range(N - 1):
            for j in range(N - 1):
                v0 = i * N + j + N * N
                v1 = v0 + 1
                v2 = (i + 1) * N + j + N * N
                v3 = v2 + 1
                faces.append([v0, v2, v1])
                face_normals.append([v0, v1, v2])
                faces.append([v1, v2, v3])
                face_normals.append([v1, v3, v2])

        # 侧面三角形（四条边界）
        # 上边界 (i=0, j=0..N-1)
        for j in range(N - 1):
            v0 = j
            v1 = j + 1
            v2 = j + N * N
            v3 = v1 + N * N
            faces.append([v0, v2, v1])
            face_normals.append([v0, v1, v2])
            faces.append([v1, v2, v3])
            face_normals.append([v1, v3, v2])

        # 下边界 (i=N-1, j=0..N-1)
        for j in range(N - 1):
            v0 = (N - 1) * N + j
            v1 = v0 + 1
            v2 = v0 + N * N
            v3 = v1 + N * N
            faces.append([v0, v1, v2])
            face_normals.append([v0, v2, v1])
            faces.append([v1, v3, v2])
            face_normals.append([v1, v2, v3])

        # 左边界 (j=0, i=0..N-1)
        for i in range(N - 1):
            v0 = i * N
            v1 = (i + 1) * N
            v2 = v0 + N * N
            v3 = v1 + N * N
            faces.append([v0, v1, v2])
            face_normals.append([v0, v2, v1])
            faces.append([v1, v3, v2])
            face_normals.append([v1, v2, v3])

        # 右边界 (j=N-1, i=0..N-1)
        for i in range(N - 1):
            v0 = i * N + (N - 1)
            v1 = (i + 1) * N + (N - 1)
            v2 = v0 + N * N
            v3 = v1 + N * N
            faces.append([v0, v2, v1])
            face_normals.append([v0, v1, v2])
            faces.append([v1, v2, v3])
            face_normals.append([v1, v3, v2])

        faces = torch.tensor(faces, dtype=torch.long)  # [F, 3]
        face_normals = torch.tensor(face_normals, dtype=torch.long)  # [F, 3]

        # 保存为 OBJ 文件
        try:
            with open(output_file, 'w') as f:
                # 写入顶点
                for v in vertices:
                    f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
                # 写入法向量
                for vn in vertex_normals:
                    f.write(f"vn {vn[0]:.6f} {vn[1]:.6f} {vn[2]:.6f}\n")
                # 写入面（OBJ 索引从 1 开始）
                for face, fn in zip(faces, face_normals):
                    f.write(f"f {face[0]+1}//{fn[0]+1} {face[1]+1}//{fn[1]+1} {face[2]+1}//{fn[2]+1}\n")
        except IOError as e:
            raise IOError(f"Failed to write to {output_file}: {str(e)}")

        return vertices, faces, vertex_normals

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise

def save_config_to_file(output_folder, filename="config.txt", readfilename="config.py"):
    """
    将config.py文件的内容直接复制到输出文件夹
    
    Args:
        output_folder: 输出文件夹路径
        filename: 配置文件名
    """
    import os
    import datetime
    
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)
    
    config_file_path = os.path.join(output_folder, filename)
    
    try:
        # 读取config.py文件内容
        with open(readfilename, 'r', encoding='utf-8') as source_file:
            config_content = source_file.read()
        
        # 写入到输出文件夹
        with open(config_file_path, 'w', encoding='utf-8') as target_file:
            # 添加时间戳
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            target_file.write(f"# Configuration snapshot saved at: {timestamp}\n")
            target_file.write(f"# This is a copy of {readfilename} at the time of training\n\n")
            target_file.write(config_content)
        
        print(f"配置文件已保存到: {config_file_path}")
        
    except FileNotFoundError:
        print(f"错误: 找不到 {readfilename} 文件")
    except Exception as e:
        print(f"保存配置文件时出错: {e}")
        
            
import numpy as np
def to_numpy(data):
    if isinstance(data, np.ndarray):
        return data  # 如果已经是 NumPy 数组，直接返回
    elif data.is_cuda:
        return data.clone().detach().cpu().numpy()  # CUDA 张量
    else:
        return data.numpy()  # CPU 张量 
    
def save_to_obj(heights, normals, nu, nv, filename):
    """
    将B样条曲面保存为OBJ文件，生成一个封闭的模型
    参数:
        heights:[N²,] 顶面顶点的高度
        normals:[N², 3] 顶面顶点的法向量
        filename: 输出的obj文件名
        nu: u方向的采样点数量
        nv: v方向的采样点数量
    """
    print("开始写入obj")
    os.makedirs(os.path.dirname(filename), exist_ok=True)  # 确保目录存在
    # 生成采样网格
    u = torch.linspace(0, 1, nu) 
    v = torch.linspace(0, 1, nv) 
    u_grid, v_grid = torch.meshgrid(u, v)
    points = torch.stack([u_grid.flatten(), v_grid.flatten()], dim=1).to(heights.device)
    
    # 将数据转换为numpy数组
    points_np = points.detach().cpu().numpy()
    top_heights_np = to_numpy(heights)
    top_normals_np = to_numpy(normals)

    z_min = min(top_heights_np)-0.01
    
    assert len(points_np) == len(top_heights_np) == len(top_normals_np)
    
    # 写入OBJ文件
    with open(filename, 'w') as f:
        # 1. 写入顶面顶点 (v)
        for i in range(len(points_np)):
            x, y = points_np[i]
            z = top_heights_np[i]
            f.write(f'v {x:.6f} {y:.6f} {z:.6f}\n')
        
        # 2. 写入底面顶点 (v) - z=z_min
        for i in range(len(points_np)):
            x, y = points_np[i]
            f.write(f'v {x:.6f} {y:.6f} {z_min:.6f}\n')
        
        # 3. 写入顶面法向量 (vn)
        for nx, ny, nz in top_normals_np:
            f.write(f'vn {nx:.6f} {ny:.6f} {nz:.6f}\n')
        
        # 4. 写入底面法向量 (vn) - 朝下
        for _ in range(len(points_np)):
            f.write('vn 0.000000 0.000000 -1.000000\n')
        
        # 5. 写入侧面法向量 (vn) - 水平向外
        side_normals = []
        for i in range(nv):
            # 左边界
            p1 = points_np[i]
            normal = np.array([-1.0, 0.0, 0.0])
            side_normals.append(normal)
            f.write(f'vn {normal[0]:.6f} {normal[1]:.6f} {normal[2]:.6f}\n')
        
            # 右边界
            p1 = points_np[i + (nu-1)*nv]
            normal = np.array([1.0, 0.0, 0.0])
            side_normals.append(normal)
            f.write(f'vn {normal[0]:.6f} {normal[1]:.6f} {normal[2]:.6f}\n')
        
        for i in range(nu):
            # 前边界
            p1 = points_np[i*nv]
            normal = np.array([0.0, -1.0, 0.0])
            side_normals.append(normal)
            f.write(f'vn {normal[0]:.6f} {normal[1]:.6f} {normal[2]:.6f}\n')
        
            # 后边界
            p1 = points_np[i*nv + nv-1]
            normal = np.array([0.0, 1.0, 0.0])
            side_normals.append(normal)
            f.write(f'vn {normal[0]:.6f} {normal[1]:.6f} {normal[2]:.6f}\n')
        
        num_points = nu * nv
        vn_offset = 1  # 法向量索引偏移
        
        # 6. 写入顶面面片 (f) - 修改顶点顺序，确保法向量朝上
        for i in range(nu-1):
            for j in range(nv-1):
                v1 = i * nv + j + 1
                v2 = i * nv + (j + 1) + 1
                v3 = (i + 1) * nv + (j + 1) + 1
                v4 = (i + 1) * nv + j + 1
                # 修改顶点顺序为逆时针
                f.write(f'f {v1}//{v1} {v4}//{v4} {v3}//{v3} {v2}//{v2}\n')
        
        # 7. 写入底面面片 (f) - 修改顶点顺序，确保法向量朝下
        for i in range(nu-1):
            for j in range(nv-1):
                v1 = i * nv + j + 1 + num_points
                v2 = i * nv + (j + 1) + 1 + num_points
                v3 = (i + 1) * nv + (j + 1) + 1 + num_points
                v4 = (i + 1) * nv + j + 1 + num_points
                vn = num_points + v1 - num_points
                # 保持顺时针顺序
                f.write(f'f {v1}//{vn} {v2}//{vn} {v3}//{vn} {v4}//{vn}\n')
        
        # 8. 写入侧面面片 (f) - 重新检查所有侧面的顶点顺序
        vn_side_start = 2 * num_points + 1
        
        # 左右侧面
        for i in range(nv-1):
            # 左侧面 (x=0)
            v1 = i + 1
            v2 = i + 2
            v3 = v2 + num_points
            v4 = v1 + num_points
            vn = vn_side_start + i*2
            # 确保逆时针顺序，使法向量朝左（-x方向）
            f.write(f'f {v1} {v2} {v3} {v4}\n')
            
            # 右侧面 (x=1)
            v1 = i + 1 + (nu-1)*nv
            v2 = i + 2 + (nu-1)*nv
            v3 = v2 + num_points
            v4 = v1 + num_points
            vn = vn_side_start + i*2 + 1
            # 确保逆时针顺序，使法向量朝右（+x方向）
            f.write(f'f {v2}//{vn} {v1}//{vn} {v4}//{vn} {v3}//{vn}\n')
        
        # 前后侧面
        vn_front_back_start = vn_side_start + 2*nv
        for i in range(nu-1):
            # 前侧面 (y=0)
            v1 = i*nv + 1
            v2 = (i+1)*nv + 1
            v3 = v2 + num_points
            v4 = v1 + num_points
            vn = vn_front_back_start + i*2
            # 确保逆时针顺序，使法向量朝前（-y方向）
            f.write(f'f {v2} {v1} {v4} {v3}\n')
            
            # 后侧面 (y=1)
            v1 = i*nv + nv
            v2 = (i+1)*nv + nv
            v3 = v2 + num_points
            v4 = v1 + num_points
            vn = vn_front_back_start + i*2 + 1
            # 确保逆时针顺序，使法向量朝后（+y方向）
            f.write(f'f {v1}//{vn} {v2}//{vn} {v3}//{vn} {v4}//{vn}\n')
        
    print(f"已保存到OBJ文件: {filename}")

from skimage.metrics import structural_similarity as ssim
def calculate_img_metrics(target_picture: torch.Tensor, our_picture: torch.Tensor, filename):
    """
    计算两个图像间的MAE、MSE和SSIM
    
    Args:
        target_picture: 目标图像tensor [H, W] 或 [1, H, W]，范围[0,1]
        real_picture: 真实图像tensor [H, W] 或 [1, H, W]，范围[0,1]
    
    Returns:
        dict: 包含所有指标的字典
    """
    # 确保输入是2D tensor
    if target_picture.dim() == 3:
        target_picture = target_picture.squeeze(0)
    if our_picture.dim() == 3:
        our_picture = our_picture.squeeze(0)
    
    # 转换到[0,255]范围并限制
    target_255 = torch.clamp(target_picture * 255.0, 0, 255)
    our_255 = torch.clamp(our_picture * 255.0, 0, 255)
    
    # 计算MAE
    mae = torch.mean(torch.abs(target_255 - our_255) / 255.0).item() 
    
    # 计算MSE
    rmse = torch.sqrt(torch.mean(((target_255 - our_255) / 255.0 )**2)).item()
    
    # 转换为numpy计算SSIM
    target_np = target_255.detach().cpu().numpy().astype(np.uint8)
    real_np = our_255.detach().cpu().numpy().astype(np.uint8)
    
    # 计算SSIM
    ssim_value = ssim(target_np, real_np, data_range=255)
    
    metrics_dict = {
        'MAE': mae,
        'RMS': rmse,
        'SSIM': ssim_value
    }
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)  # 确保目录存在
    with open(filename, 'w') as f:
        f.write(f"MAE: {mae:.6f}\n")
        f.write(f"RMS: {rmse:.6f}\n")
        f.write(f"SSIM: {ssim_value:.6f}\n")
    