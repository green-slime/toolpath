import matplotlib.pyplot as plt
import os
import numpy as np
import torch
import config as cfg

def to_numpy(data):
    if isinstance(data, np.ndarray):
        return data  # 如果已经是 NumPy 数组，直接返回
    elif data.is_cuda:
        return data.clone().detach().cpu().numpy()  # CUDA 张量
    else:
        return data.numpy()  # CPU 张量 
    
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
    
def drawPairPos(points_cal, points_fixed, epoch, filename="output/pair_pos.png"):
    """
    points_cal: 计算点
    points_fixed: 固定目标
    """
    # 蓝点1为迭代计算点，红点2为固定目标
    print("绘制 PairPos 图像中...")
    os.makedirs(os.path.dirname(filename),exist_ok=True)
    points_cal_np = to_numpy(points_cal)
    points_fixed_np = to_numpy(points_fixed)
    # 创建图形
    plt.figure(figsize=(8, 8))
    # 绘制点集
    plt.scatter(points_cal_np[:, 0], points_cal_np[:, 1], c='blue', label='proj_cal', s=5)
    plt.scatter(points_fixed_np[:, 0], points_fixed_np[:, 1], c='red', label='proj_fixed', s=5)

    """ # 绘制对应索引的连线
    for i in range(points_cal.shape[0]):
        plt.plot([points_cal_np[i, 0], points_fixed_np[i, 0]], 
                [points_cal_np[i, 1], points_fixed_np[i, 1]], 
                'g--', alpha=0.5)  # 绿色虚线连接 """

    # 添加标签和图例
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.1))
    plt.title(f'Epoch: {epoch}')
    plt.grid(True)
    plt.savefig(filename)
    plt.close()
    
def drawScatterPos(points, filename="output/scatter_pos.png"):
    """
    绘制散点图
    points: 形状为 (N, 2) 的张量或数组
    """
    print("绘制 ScatterPos 图像中...")
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    points_np = to_numpy(points)
    
    plt.figure(figsize=(8, 8))
    plt.scatter(points_np[:, 0], points_np[:, 1], c='red', s=1)
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Scatter Points')
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

def drawHeatMap(data, path, title="render_result"):
    if os.path.dirname(path)!='':
        os.makedirs(os.path.dirname(path), exist_ok=True)
    data = to_numpy(data)
    plt.figure(figsize=(8, 8))  # 设置图形大小
    plt.imshow(data, cmap='jet')  # 使用 'jet' 颜色映射（常见热力图风格）
    plt.colorbar()  # 添加颜色条
    plt.title(title)  # 设置标题
    plt.savefig(path)
    plt.close()

def drawTwoHeatMap(target, result, path, title1="Target", title2="Result"):
    from matplotlib.colors import Normalize
    if os.path.dirname(path)!='':
        os.makedirs(os.path.dirname(path), exist_ok=True)
    target = to_numpy(target)
    result = to_numpy(result)

    global_min = min(target.min(), result.min())
    global_max = max(target.max(), result.max())

    # 2. 创建统一的范围标准化器
    norm = Normalize(vmin=global_min, vmax=global_max)

    # 3. 绘制图像（共享相同的norm对象）
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), 
                              constrained_layout=True)

    im1 = ax1.imshow(target, cmap='jet', norm=norm)
    ax1.set_title(title1)
    im2 = ax2.imshow(result, cmap='jet', norm=norm)
    ax2.set_title(title2)

    # 4. 添加共享的颜色条
    cbar = fig.colorbar(im2, ax=[ax1, ax2], location='right', shrink=0.6)
    cbar.set_label('Value Scale')

    plt.savefig(path)
    plt.close()

def drawThreeMap(original, target, path, title_original="Original", title_target="Target"):
    """
    绘制原图、目标图和差异热力图
    
    Parameters:
    -----------
    original : tensor or array
        原图张量
    target : tensor or array  
        目标张量
    path : str
        保存路径
    """
    if os.path.dirname(path) != '':
        os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # 转换为numpy数组
    original_np = to_numpy(original)
    target_np = to_numpy(target)
    
    # 乘以255并限制在[0,255]范围内
    original_scaled = np.clip(original_np * 255, 0, 255).astype(np.uint8)
    target_scaled = np.clip(target_np * 255, 0, 255).astype(np.uint8)
    
    # 计算差异
    diff = original_np - target_np
    
    # 创建子图
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # 显示原图（灰度图）
    ax1.imshow(original_scaled, cmap='gray', vmin=0, vmax=255)
    ax1.set_title(title_original)
    ax1.axis('off')
    
    # 显示目标图（灰度图）
    ax2.imshow(target_scaled, cmap='gray', vmin=0, vmax=255)
    ax2.set_title(title_target)
    ax2.axis('off')
    
    # 显示差异热力图
    im3 = ax3.imshow(diff, cmap='jet')
    ax3.set_title('Difference Heatmap')
    ax3.axis('off')
    
    # 添加颜色条
    plt.colorbar(im3, ax=ax3, shrink=0.6)
    
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    
def drawFiveMap(target, init_render_result, final_render_result, path):
    """
    绘制五张对比图：目标图、初始结果、最终结果、初始MAE、最终MAE
    使用2x3布局，第一列跨行显示目标图
    
    Parameters:
    -----------
    target : tensor or array
        目标图像张量
    init_render_result : tensor or array
        初始渲染结果张量
    final_render_result : tensor or array
        最终渲染结果张量
    path : str
        保存路径
    """
    if os.path.dirname(path) != '':
        os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # 转换为numpy数组
    target_np = to_numpy(target)
    init_np = to_numpy(init_render_result)
    final_np = to_numpy(final_render_result)
    
    # 乘以255并限制在[0,255]范围内用于显示
    target_scaled = np.clip(target_np * 255, 0, 255).astype(np.uint8)
    init_scaled = np.clip(init_np * 255, 0, 255).astype(np.uint8)
    final_scaled = np.clip(final_np * 255, 0, 255).astype(np.uint8)
    
    # 计算MAE（绝对误差）
    init_mae = np.abs(target_scaled.astype(float) - init_scaled.astype(float)) / target_scaled.astype(float).max()
    
    final_mae = np.abs(target_scaled.astype(float) - final_scaled.astype(float)) / target_scaled.astype(float).max()
    
    # 计算统一的MAE范围（用于两个热力图）
    global_mae_max = max(init_mae.max(), final_mae.max())
    
    # 创建2x3布局的图形
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle('Rendering Results Comparison', fontsize=16, fontweight='bold')
    
    # 第一列跨行：目标图像 (位置1和4)
    ax1 = fig.add_subplot(2, 3, (1, 4))
    ax1.imshow(target_scaled, cmap='gray', vmin=0, vmax=255)
    ax1.set_title('Target Image', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # 第一行：初始结果和初始MAE
    # 位置2：初始渲染结果
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.imshow(init_scaled, cmap='gray', vmin=0, vmax=255)
    ax2.set_title('Initial Render Result', fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    # 位置3：初始MAE热力图（不单独添加colorbar）
    ax3 = fig.add_subplot(2, 3, 3)
    im3 = ax3.imshow(init_mae, cmap='jet', 
                     vmin=0, vmax=global_mae_max)
    ax3.set_title('Initial MAE', fontsize=14, fontweight='bold')
    ax3.axis('off')

    # 第二行：最终结果和最终MAE
    # 位置5：最终渲染结果
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.imshow(final_scaled, cmap='gray', vmin=0, vmax=255)
    ax5.set_title('Final Render Result', fontsize=14, fontweight='bold')
    ax5.axis('off')
    
    # 位置6：最终MAE热力图（不单独添加colorbar）
    ax6 = fig.add_subplot(2, 3, 6)
    im6 = ax6.imshow(final_mae, cmap='jet', 
                     vmin=0, vmax=global_mae_max)
    ax6.set_title('Final MAE', fontsize=14, fontweight='bold')
    ax6.axis('off')

    # 在整个图的右侧添加一条共享colorbar
    fig.colorbar(im3, ax=[ax3, ax6], shrink=0.6, aspect=30, pad=0.08)
    
    #plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    
    

def visualize_surface_vectors_3d(surface_points, direction_vectors, 
                                grid_size=None, filename='output/surface_vectors_3d.png', scale=0.02, 
                                arrow_color='red', point_color='blue',
                                surface_alpha=0.3, arrow_alpha=1.0,
                                figsize=(12, 8)):
    """
    使用matplotlib 3D箭头图可视化曲面上的点和对应的方向场
    
    Parameters:
    -----------
    surface_points : array-like, shape (N^2, 3)
        曲面上的采样点坐标 [x, y, z]
    direction_vectors : array-like, shape (N^2, 3)
        每个点对应的方向向量 [u, v, w]
    grid_size : int, optional
        如果已知网格大小N，可以重构为网格形式显示表面
    scale : float, default=1.0
        箭头长度的缩放因子
    arrow_color : str, default='red'
        箭头颜色
    point_color : str, default='blue'
        点的颜色
    surface_alpha : float, default=0.3
        表面透明度
    arrow_alpha : float, default=0.8
        箭头透明度
    figsize : tuple, default=(12, 8)
        图形大小
    """
    
    # 转换为numpy数组
    points = to_numpy(surface_points)
    vectors = to_numpy(direction_vectors)
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # 验证输入数据
    if points.shape[1] != 3:
        raise ValueError("surface_points应该是形状为(N^2, 3)的数组")
    if vectors.shape[1] != 3:
        raise ValueError("direction_vectors应该是形状为(N^2, 3)的数组")
    if points.shape[0] != vectors.shape[0]:
        raise ValueError("点的数量和向量的数量必须相同")
    
    # 提取坐标分量
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    u, v, w = vectors[:, 0], vectors[:, 1], vectors[:, 2]
    
    # 创建图形
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # 如果提供了网格大小，尝试重构表面
    if grid_size is not None:
        n_points = points.shape[0]
        expected_points = grid_size * grid_size
        
        if n_points == expected_points:
            # 重构为网格形式
            X = x.reshape(grid_size, grid_size)
            Y = y.reshape(grid_size, grid_size)
            Z = z.reshape(grid_size, grid_size)
            
            # 绘制表面
            ax.plot_surface(X, Y, Z, alpha=surface_alpha, 
                          color='lightblue', edgecolor='none')
            
            # 稀疏采样显示箭头（避免过于密集）
            N = points.shape[0]
            sample_ratio = 0.03 * 0.03
            sample_size = max(1, int(N * sample_ratio))
            indices = np.random.choice(N, size=sample_size, replace=False)
            
            ax.quiver(x[indices], y[indices], z[indices], 
                     u[indices], v[indices], w[indices],
                     color=arrow_color, alpha=arrow_alpha, linewidth=1.0,
                     length=scale, normalize=True, arrow_length_ratio=0.03)
            
            ax.set_title(f'Surface with Vector Field (Grid: {grid_size}x{grid_size})')
        else:
            print(f"警告：网格大小{grid_size}^2={expected_points}与点数{n_points}不匹配")
    
    # 如果没有网格大小或网格重构失败，显示所有点
    if grid_size is None or points.shape[0] != grid_size * grid_size:
        # 显示采样点
        ax.scatter(x, y, z, c=point_color, s=20, alpha=0.6)
        
        # 如果点太多，进行稀疏采样
        n_points = len(x)
        if n_points > 200:
            step = n_points // 200
            indices = np.arange(0, n_points, step)
            print(f"点数过多({n_points})，使用稀疏采样显示箭头({len(indices)}个)")
        else:
            indices = np.arange(n_points)
        
        # 绘制方向向量
        ax.quiver(x[indices], y[indices], z[indices], 
                 u[indices], v[indices], w[indices],
                 color=arrow_color, alpha=arrow_alpha, 
                 length=scale, normalize=True)
        
        ax.set_title(f'Surface Points with Vector Field ({n_points} points)')
    
    # 设置坐标轴
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # 设置等比例坐标轴
    max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
    mid_x = (x.max()+x.min()) * 0.5
    mid_y = (y.max()+y.min()) * 0.5
    mid_z = (z.max()+z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    #ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.tight_layout()
    #plt.show()
    plt.savefig(filename)
    plt.close()
    
    return fig, ax

from mayavi import mlab

def visualize_surface_vectors_3d_with_mayavi(points, directions):
    """
    Visualize 3D points and their corresponding direction vectors.
    
    Parameters:
    -----------
    points : numpy.ndarray
        Array of shape (N, 3) containing 3D point coordinates
    directions : numpy.ndarray
        Array of shape (N, 3) containing direction vectors for each point
    """
    # Make sure inputs are numpy arrays
    points = to_numpy(points)
    directions = to_numpy(directions)
    
    sample_ratio = 0.01
    
    if sample_ratio < 1.0:
        N = points.shape[0]
        sample_size = max(1, int(N * sample_ratio))
        indices = np.random.choice(N, size=sample_size, replace=False)
        points = points[indices]
        directions = directions[indices]
    
    # Create figure
    fig = mlab.figure(bgcolor=(1, 1, 1), size=(800, 600))
    
    # Plot points
    pts = mlab.points3d(
        points[:, 0], points[:, 1], points[:, 2],
        scale_factor=0.05,  # Adjust size of points
        color=(0, 0, 1),    # Blue color for points
        resolution=8,        # Resolution of sphere
        opacity=0.5
    )
    
    # Plot direction vectors
    vectors = mlab.quiver3d(
        points[:, 0], points[:, 1], points[:, 2],
        directions[:, 0], directions[:, 1], directions[:, 2],
        scale_factor=0.1,   # Adjust length of arrows
        color=(1, 0, 0),    # Red color for vectors
        line_width=2.0,     # Width of arrows
        scale_mode='vector' # Scale according to vector magnitude
    )
    
    # Add orientation axes for reference
    mlab.orientation_axes()
    
    # Display the visualization
    mlab.show()
    mlab.savefig("surface_vectors_3d_mayavi.png")

def visualize_paths_and_surface(surface_points, path_points, 
                               grid_size=None, filename='output/paths_and_surface.png',
                               path_sample_ratio=0.02, surface_alpha=0.3,
                               path_colors=['red', 'blue', 'green', 'orange', 'purple'],
                               figsize=(12, 8)):
    """
    可视化路径和曲面
    
    Parameters:
    -----------
    surface_points : array-like, shape (N^2, 3)
        曲面上的采样点坐标 [x, y, z]
    path_points : array-like, shape (path_num, path_len, 3)
        路径点坐标
    grid_size : int, optional
        网格大小，用于重构曲面
    path_sample_ratio : float, default=0.1
        路径采样比例，0.1表示显示10%的路径
    surface_alpha : float, default=0.3
        表面透明度
    path_colors : list
        路径颜色列表
    """
    
    # 转换为numpy数组
    surface_pts = to_numpy(surface_points)
    paths = to_numpy(path_points)
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # 创建图形
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制曲面
    if grid_size is not None:
        n_points = surface_pts.shape[0]
        expected_points = grid_size * grid_size
        
        if n_points == expected_points:
            x, y, z = surface_pts[:, 0], surface_pts[:, 1], surface_pts[:, 2]
            X = x.reshape(grid_size, grid_size)
            Y = y.reshape(grid_size, grid_size)
            Z = z.reshape(grid_size, grid_size)
            
            # 绘制表面
            ax.plot_surface(X, Y, Z, alpha=surface_alpha, 
                          color='lightblue', edgecolor='gray', linewidth=0.1)
    else:
        # 如果没有网格信息，绘制散点
        x, y, z = surface_pts[:, 0], surface_pts[:, 1], surface_pts[:, 2]
        ax.scatter(x, y, z, c='lightblue', s=10, alpha=0.5)
    
    # 绘制路径
    path_num, path_len, _ = paths.shape
    sample_num = max(1, int(path_num * path_sample_ratio))
    
    # 均匀采样路径索引
    if sample_num >= path_num:
        selected_paths = range(path_num)
    else:
        step = path_num // sample_num
        selected_paths = range(0, path_num, step)[:sample_num]
    
    print(f"显示 {len(selected_paths)} 条路径（共 {path_num} 条）")
    
    for i, path_idx in enumerate(selected_paths):
        path = paths[path_idx]  # shape: (path_len, 3)
        color = path_colors[i % len(path_colors)]
        
        # 绘制路径线
        ax.plot(path[:, 0], path[:, 1], path[:, 2]-cfg.R, 
                color=color, linewidth=2, alpha=0.8, 
                label=f'Path {path_idx}')
        
        # 在路径端点添加标记
        ax.scatter(path[0, 0], path[0, 1], path[0, 2]-cfg.R, 
                  color=color, s=10, marker='o', alpha=1.0)  # 起点
        ax.scatter(path[-1, 0], path[-1, 1], path[-1, 2]-cfg.R, 
                  color=color, s=10, marker='s', alpha=1.0)  # 终点
    
    # 设置坐标轴
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Paths and Surface (Showing {len(selected_paths)}/{path_num} paths)')
    
    # 添加图例（如果路径不太多）
    if len(selected_paths) <= 10:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 设置等比例坐标轴
    all_points = np.vstack([surface_pts, paths.reshape(-1, 3)])
    max_range = np.array([all_points[:, 0].max() - all_points[:, 0].min(),
                         all_points[:, 1].max() - all_points[:, 1].min(),
                         all_points[:, 2].max() - all_points[:, 2].min()]).max() / 2.0
    mid_x = (all_points[:, 0].max() + all_points[:, 0].min()) * 0.5
    mid_y = (all_points[:, 1].max() + all_points[:, 1].min()) * 0.5
    mid_z = (all_points[:, 2].max() + all_points[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    #ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    return fig, ax