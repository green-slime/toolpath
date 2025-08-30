import os
import torch
import shutil
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def load_dinov2_model(model_path, device='cuda', force_download=False):
    """
    加载DINOv2模型，支持本地缓存和自动下载
    参数：
        model_path: 自定义模型存储路径（目录）
        device: 目标设备
        force_download: 强制重新下载模型
    返回：
        DINOv2模型实例
    """
    # 确保目录存在
    model_dir = Path(model_path)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # 定义标准缓存路径（PyTorch Hub默认路径）
    hub_dir = torch.hub.get_dir()
    repo_dir = os.path.join(hub_dir, "facebookresearch_dinov2_main")
    
    try:
        # 优先尝试加载本地缓存模型
        if not force_download and (model_dir / "dinov2_vits14.pth").exists():
            print(f"Loading model from local cache: {model_dir}")
            model = torch.hub.load('models/dinov2/hub_cache', 'dinov2_vits14', source='local')
            return model.to(device)
            
        # 如果强制下载或本地不存在，则从Hub下载
        print("Downloading model from torch.hub...")
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', pretrained=True)
        
        # 保存完整模型结构+权重到自定义路径
        torch.save(model.state_dict(), model_dir / "dinov2_vits14.pth")
        print(f"Model saved to {model_dir}")
        
        # 额外保存整个仓库配置（可选）
        if os.path.exists(repo_dir):
            shutil.copytree(repo_dir, model_dir / "hub_cache", dirs_exist_ok=True)
            
        return model.to(device)
        
    except Exception as e:
        # 异常处理（网络问题/权限问题等）
        raise RuntimeError(f"Failed to load model: {str(e)}") from e

def to_numpy(data):
    if isinstance(data, np.ndarray):
        return data  # 如果已经是 NumPy 数组，直接返回
    elif data.is_cuda:
        return data.clone().detach().cpu().numpy()  # CUDA 张量
    else:
        return data.numpy()  # CPU 张量 
    
def drawHeatMap(data, path, title="render_diff visualization"):
    if os.path.dirname(path)!='':
        os.makedirs(os.path.dirname(path), exist_ok=True)
    data = to_numpy(data)
    plt.figure(figsize=(8, 8))  # 设置图形大小
    plt.imshow(data, cmap='jet')  # 使用 'jet' 颜色映射（常见热力图风格）
    plt.colorbar()  # 添加颜色条
    plt.title(title)  # 设置标题
    plt.savefig(path)
    plt.close()

def drawDelaunyTriangulation(xy_data, triangles:torch.tensor, epoch, folder='triangulation'):
    if folder!='':
        os.makedirs(folder,exist_ok=True)
    xy_data = to_numpy(xy_data)
    triangles = to_numpy(triangles)
    plt.figure(figsize=(8, 8))
    # 绘制原始点
    plt.scatter(xy_data[:, 0], xy_data[:, 1], c='blue', label='Points')
    # 绘制三角形
    plt.triplot(xy_data[:, 0], xy_data[:, 1], triangles, color='red', label='Delaunay Triangulation')
    plt.title(f"Epoch: {epoch}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True)
    plt.savefig(folder + f"/delaunay_triangulation{epoch}.png")
    plt.close()

def drawLossCurve(losses, path='loss.png'):
    # 可视化
    plt.figure(figsize=(10, 6))
    plt.plot(losses, marker='o', linestyle='-', color='b', label='Training Loss')
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.savefig(path)
    plt.close()
    
def drawPairPos(points1, points2, epoch, folder="yz"):
    """
    points1: 计算点
    points2: 固定目标
    """
    # 蓝点1为迭代计算点，红点2为固定目标
    os.makedirs(folder,exist_ok=True)
    points1_np = to_numpy(points1)
    points2_np = to_numpy(points2)
    # 创建图形
    plt.figure(figsize=(8, 8))
    # 绘制点集
    plt.scatter(points1_np[:, 0], points1_np[:, 1], c='blue', label='proj_cal', s=5)
    plt.scatter(points2_np[:, 0], points2_np[:, 1], c='red', label='proj_fixed', s=5)

    # 绘制对应索引的连线
    for i in range(points1.shape[0]):
        plt.plot([points1_np[i, 0], points2_np[i, 0]], 
                [points1_np[i, 1], points2_np[i, 1]], 
                'g--', alpha=0.5)  # 绿色虚线连接

    # 添加标签和图例
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.1))
    plt.title('Two Point Sets with Corresponding Lines')
    plt.grid(True)
    plt.savefig(folder + f"/epoch{epoch}.png")
    plt.close()

def draw_3dirs(refracted_dir, normals, incident_dir):
    refracted_dir_np = refracted_dir.detach().cpu().numpy()
    normals_np = normals.detach().cpu().numpy()
    incident_dir_np = incident_dir.detach().cpu().numpy()

    # 选择部分样本
    num_samples = 1
    indices = np.random.choice(refracted_dir_np.shape[0], min(num_samples, refracted_dir_np.shape[0]), replace=False)
    refracted_dir_sample = refracted_dir_np[indices]
    normals_sample = normals_np[indices]

    # 创建 3D 图形
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制入射方向 (从原点开始，固定方向)
    ax.quiver(0, 0, 0, incident_dir_np[0], incident_dir_np[1], incident_dir_np[2],
            color='b', label='Incident Direction', linewidth=2)

    # 绘制折射方向 (从原点开始)
    for i in range(len(indices)):
        ax.quiver(0, 0, 0, refracted_dir_sample[i, 0], refracted_dir_sample[i, 1], refracted_dir_sample[i, 2],
                color='r', alpha=0.5)

    # 绘制法向 (从原点开始)
    for i in range(len(indices)):
        ax.quiver(0, 0, 0, normals_sample[i, 0], normals_sample[i, 1], normals_sample[i, 2],
                color='g', alpha=0.5)

    # 设置图例和标签
    ax.plot([], [], [], color='r', label='Refracted Direction', linewidth=2)
    ax.plot([], [], [], color='g', label='Normal', linewidth=2)
    ax.legend()

    # 设置坐标轴范围
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Incident, Refracted Directions, and Normals')
    
    plt.savefig('3dirs.png')
    
    
def draw_2heights(z_fixed, z, epoch):
    # 控制点连接高度
    from config import nu,nv,control_points_num as cp_num
    os.makedirs("2heights",exist_ok=True)
    # 数据同上
    x = np.linspace(0, 1, cp_num)
    y = np.linspace(0, 1, cp_num)
    X, Y = np.meshgrid(x, y)
    Z1 = to_numpy(z_fixed)
    Z2 = to_numpy(z)
    Z1 = Z1.reshape(X.shape)
    Z2 = Z2.reshape(X.shape)

    # 可视化 - 两个 3D 表面
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # 高度场 1（蓝色，半透明）
    surf1 = ax.plot_surface(X, Y, Z1, cmap='Blues', alpha=0.5, label='z_fixed')

    # 高度场 2（红色，半透明）
    surf2 = ax.plot_surface(X, Y, Z2, cmap='Reds', alpha=0.5, label='z')
    #ax.set_zlim(0, 1)

    # 设置标题和标签
    ax.set_title('Two Height Fields (Surfaces)')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # 图例（手动添加）
    ax.legend(['Height Field 1', 'Height Field 2'])

    plt.savefig(f'2heights/epoch{epoch}.png')
    plt.close()       
    
def read_off_to_tensor(filename):
    vertices = []
    with open(filename, 'r') as file:
        # 读取第一行
        header = file.readline().strip()
        if header != "OFF":
            print("不是有效的 OFF 文件")
            return None

        # 读取第二行：顶点数、面数、边数
        counts = file.readline().strip().split()
        vertex_count = int(counts[0])

        # 读取顶点坐标
        for _ in range(vertex_count):
            line = file.readline().strip().split()
            x, y, z = map(float, line[:3])  # 只取前三个数作为坐标
            vertices.append([x, y, z])

    # 转换为 PyTorch Tensor
    vertex_tensor = torch.tensor(vertices, dtype=torch.float32)
    return vertex_tensor

import struct

def read_vectors_from_binary(filename):
    vertices = []
    with open(filename, 'rb') as file:
        # 读取向量大小 (size_t 通常是 8 字节)
        size_bytes = file.read(8)
        num_vectors = struct.unpack('Q', size_bytes)[0]  # 'Q' 表示 unsigned long long

        # 读取每个 Vec2 的 x, y (每个 double 是 8 字节)
        for _ in range(num_vectors):
            x_bytes = file.read(8)
            y_bytes = file.read(8)
            x = struct.unpack('d', x_bytes)[0]  # 'd' 表示 double
            y = struct.unpack('d', y_bytes)[0]
            vertices.append([x, y])

    # 转换为 PyTorch Tensor
    return torch.tensor(vertices, dtype=torch.float32)

from PIL import Image
import config as cfg

def resize_and_save_image(input_path, size, output_path=None):
    """
    读取图片，调整到指定大小并保存。

    Args:
        input_path (str): 输入图片路径。
        output_path (str): 输出图片路径。默认：/data/wzr/2025/img/{name}_{size[0]}.png
        size (tuple): 目标大小 (width, height)。
    """
    if isinstance(size, int):
        size = (size, size)
        
    if output_path is None:
        name = Path(input_path).stem
        output_path = f"{cfg.project_dir}/img/{name}_{size[0]}.png"
    # 读取图片
    
    img = Image.open(input_path)

    # 调整大小
    img_resized = img.resize(size, Image.Resampling.LANCZOS)  # LANCZOS 提供高质量插值

    # 保存图片
    img_resized.save(output_path)
    print(f"Image saved to {output_path} with size {size}")
    return output_path
    
def visualize_points(tensor, N, boundary_color='red', interior_color='blue'):
    """
    可视化 [N**2, 2] 张量的二维点分布，边界点用不同颜色标识。

    Args:
        tensor (torch.Tensor): 形状为 [N**2, 2] 的张量，表示二维点坐标。
        N (int): 网格的大小（N x N）。
        boundary_color (str): 边界点颜色，默认 'red'。
        interior_color (str): 内部点颜色，默认 'blue'。
    """
    # 确保张量形状正确
    assert tensor.shape == (N * N, 2), f"Expected shape [{N**2}, 2], got {tensor.shape}"

    # 转换为 NumPy 数组
    points = tensor.detach().cpu().numpy()

    # 创建掩码标识边界点
    boundary_mask = np.zeros(N * N, dtype=bool)
    for i in range(N * N):
        row = i // N
        col = i % N
        if row == 0 or row == N - 1 or col == 0 or col == N - 1:
            boundary_mask[i] = True

    # 分离边界点和内部点
    boundary_points = points[boundary_mask]
    interior_points = points[~boundary_mask]

    # 创建散点图
    plt.figure(figsize=(8, 8))
    plt.scatter(interior_points[:, 0], interior_points[:, 1], c=interior_color, label='Interior Points', s=5)
    plt.scatter(boundary_points[:, 0], boundary_points[:, 1], c=boundary_color, label='Boundary Points', s=5)

    # 添加标签和图例
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'Distribution of {N**2} Points (N={N})')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')  # 保持 x 和 y 轴比例相等
    plt.show()
    plt.savefig('visualize_points.png')

def gamma_correction(image_path, output_path, gamma):
    # 读入图片
    img = Image.open(image_path)
    
    # 转换为 numpy 数组并归一化到 [0, 1]
    img_array = np.array(img) / 255.0
    
    # 应用 Gamma 矫正
    gamma_corrected = np.power(img_array, gamma)
    
    # 转换回 [0, 255] 并确保数据类型为 uint8
    gamma_corrected = (gamma_corrected * 255).astype(np.uint8)
    
    # 创建新的图像并保存
    img_corrected = Image.fromarray(gamma_corrected)
    img_corrected.save(output_path)
    print(f"已保存 Gamma 矫正后的图片到: {output_path}")
    

if __name__ == "__main__":
    resize_and_save_image('designed_img.png', (200,200))
# 示例使用

if __name__ == "__main__":
    # 创建示例 [N**2, 2] 张量（假设是 N x N 网格）
    N = 5
    x = torch.linspace(0, 1, N)
    y = torch.linspace(0, 1, N)
    X, Y = torch.meshgrid(x, y, indexing='ij')  # 生成网格
    tensor = torch.stack([X.flatten(), Y.flatten()], dim=1)  # [N**2, 2]

    # 添加一些随机扰动（可选）
    tensor += torch.randn(N * N, 2) * 0.05

    # 可视化
    visualize_points(tensor, N)

    
""" if __name__ == "__main__":
    from time import time
    time1 = time()
    tensor = read_vectors_from_binary("/data/wzr/otmap/build/vectors.bin")
    print(f'use time: {time()-time1:.2f}s')
    print("Tensor:\n", tensor)
    print("Shape:", tensor.shape) """
       
""" # 使用示例
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    custom_model_path = "./models/dinov2"  # 自定义存储路径
    
    # 首次运行会下载并保存模型
    model = load_dinov2_model(custom_model_path, device)
    
    # 再次运行会直接加载本地模型
    model_reloaded = load_dinov2_model(custom_model_path, device) """