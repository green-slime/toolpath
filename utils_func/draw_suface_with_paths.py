import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from utils import to_numpy

def visualize_surface_with_paths(surface_points, path_points_list, 
                                grid_size=None, filename='output/surface_with_paths.png', 
                                surface_alpha=0.7, path_colors=None, path_widths=None,
                                figsize=(12, 10), view_angle=(30, 45)):
    """
    绘制3D曲面并在其上显示多条路径
    
    Parameters:
    -----------
    surface_points : array-like, shape (N^2, 3)
        曲面上的采样点坐标 [x, y, z]
    path_points_list : list of arrays
        路径点列表，每个路径是形状为 (M, 3) 的数组
    grid_size : int, optional
        如果已知网格大小N，可以重构为网格形式显示表面
    surface_alpha : float, default=0.7
        表面透明度
    path_colors : list, optional
        每条路径的颜色列表，如果为None则自动分配
    path_widths : list, optional
        每条路径的线宽列表，如果为None则使用默认值
    figsize : tuple, default=(12, 10)
        图形大小
    view_angle : tuple, default=(30, 45)
        视角 (elevation, azimuth)
    """
    
    # 转换为numpy数组
    surface_pts = to_numpy(surface_points)
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # 验证输入数据
    if surface_pts.shape[1] != 3:
        raise ValueError("surface_points应该是形状为(N^2, 3)的数组")
    
    # 提取坐标分量
    x, y, z = surface_pts[:, 0], surface_pts[:, 1], surface_pts[:, 2]
    
    # 创建图形
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制曲面
    if grid_size is not None:
        n_points = surface_pts.shape[0]
        expected_points = grid_size * grid_size
        
        if n_points == expected_points:
            # 重构为网格形式
            X = x.reshape(grid_size, grid_size)
            Y = y.reshape(grid_size, grid_size)
            Z = z.reshape(grid_size, grid_size)
            
            # 绘制表面
            surf = ax.plot_surface(X, Y, Z, alpha=surface_alpha, 
                                 cmap='viridis', edgecolor='none',
                                 linewidth=0, antialiased=True)
            
            print(f"绘制网格曲面: {grid_size}x{grid_size}")
        else:
            print(f"警告：网格大小{grid_size}^2={expected_points}与点数{n_points}不匹配，使用散点显示")
            ax.scatter(x, y, z, c=z, cmap='viridis', s=1, alpha=surface_alpha)
    else:
        # 使用散点显示曲面
        ax.scatter(x, y, z, c=z, cmap='viridis', s=1, alpha=surface_alpha)
        print(f"绘制散点曲面: {len(x)} 个点")
    
    # 设置默认路径颜色和线宽
    num_paths = len(path_points_list)
    if path_colors is None:
        # 使用不同颜色
        colors = plt.cm.tab10(np.linspace(0, 1, num_paths))
        path_colors = [colors[i] for i in range(num_paths)]
    
    if path_widths is None:
        path_widths = [3.0] * num_paths
    
    # 绘制路径
    for i, path_points in enumerate(path_points_list):
        path_pts = to_numpy(path_points)
        
        if path_pts.shape[1] != 3:
            print(f"警告: 路径 {i} 的维度不正确，跳过")
            continue
        
        px, py, pz = path_pts[:, 0], path_pts[:, 1], path_pts[:, 2]
        
        # 绘制路径线条
        ax.plot(px, py, pz, color=path_colors[i], linewidth=path_widths[i],
               label=f'Path {i+1}', alpha=0.9)
        
        # 可选：在路径起点和终点添加标记
        ax.scatter(px[0], py[0], pz[0], color=path_colors[i], s=100, 
                  marker='o', edgecolor='black', linewidth=1, alpha=1.0)
        ax.scatter(px[-1], py[-1], pz[-1], color=path_colors[i], s=100, 
                  marker='s', edgecolor='black', linewidth=1, alpha=1.0)
    
    # 设置坐标轴
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # 设置视角
    ax.view_init(elev=view_angle[0], azim=view_angle[1])
    
    # 添加图例
    if num_paths <= 10:  # 只有路径数量不多时才显示图例
        ax.legend(loc='upper left', bbox_to_anchor=(0, 1))
    
    # 设置标题
    ax.set_title(f'Surface with {num_paths} Path(s)', fontsize=14, fontweight='bold')
    
    # 设置等比例坐标轴（可选）
    max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
    mid_x = (x.max()+x.min()) * 0.5
    mid_y = (y.max()+y.min()) * 0.5
    mid_z = (z.max()+z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"曲面和路径图已保存到: {filename}")
    
    return fig, ax

def visualize_surface_with_toolpaths(surface_points, toolpath_data, 
                                   grid_size=None, filename='output/surface_with_toolpaths.png',
                                   surface_alpha=0.6, show_tool_contacts=True,
                                   contact_point_size=10, figsize=(14, 10)):
    """
    专门用于绘制曲面和刀具路径的函数
    
    Parameters:
    -----------
    surface_points : array-like, shape (N^2, 3)
        曲面上的采样点坐标
    toolpath_data : list or dict
        刀具路径数据，可以是:
        - list: 路径点列表
        - dict: {'paths': [...], 'tool_radius': float, 'cutting_direction': [...]}
    """
    
    # 处理刀具路径数据
    if isinstance(toolpath_data, dict):
        path_points_list = toolpath_data.get('paths', [])
        tool_radius = toolpath_data.get('tool_radius', 0.0)
        cutting_dirs = toolpath_data.get('cutting_direction', None)
    else:
        path_points_list = toolpath_data
        tool_radius = 0.0
        cutting_dirs = None
    
    # 转换为numpy数组
    surface_pts = to_numpy(surface_points)
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # 创建图形
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制曲面（使用更适合工程图的配色）
    x, y, z = surface_pts[:, 0], surface_pts[:, 1], surface_pts[:, 2]
    
    if grid_size is not None and surface_pts.shape[0] == grid_size * grid_size:
        X = x.reshape(grid_size, grid_size)
        Y = y.reshape(grid_size, grid_size)
        Z = z.reshape(grid_size, grid_size)
        
        surf = ax.plot_surface(X, Y, Z, alpha=surface_alpha, 
                             cmap='coolwarm', edgecolor='none',
                             linewidth=0, antialiased=True)
        
        # 添加颜色条
        fig.colorbar(surf, ax=ax, shrink=0.6, aspect=20)
    else:
        ax.scatter(x, y, z, c=z, cmap='coolwarm', s=2, alpha=surface_alpha)
    
    # 绘制刀具路径
    path_colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    
    for i, path_points in enumerate(path_points_list):
        path_pts = to_numpy(path_points)
        
        if path_pts.shape[1] < 3:
            continue
            
        px, py, pz = path_pts[:, 0], path_pts[:, 1], path_pts[:, 2]
        color = path_colors[i % len(path_colors)]
        
        # 绘制刀具中心路径
        ax.plot(px, py, pz, color=color, linewidth=2.5, 
               label=f'Toolpath {i+1}', alpha=0.8)
        
        # 如果显示刀具接触点且有工具半径信息
        if show_tool_contacts and tool_radius > 0:
            # 计算接触点（假设垂直向下接触）
            contact_z = pz - tool_radius
            ax.plot(px, py, contact_z, color=color, linewidth=1.5, 
                   alpha=0.6, linestyle='--', label=f'Contact {i+1}')
            
            # 显示部分接触点
            step = max(1, len(px) // 20)  # 每20个点显示一个
            ax.scatter(px[::step], py[::step], contact_z[::step], 
                      color=color, s=contact_point_size, alpha=0.7, marker='.')
        
        # 标记起点和终点
        ax.scatter(px[0], py[0], pz[0], color=color, s=80, 
                  marker='o', edgecolor='black', linewidth=1.5, alpha=1.0)
        ax.scatter(px[-1], py[-1], pz[-1], color=color, s=80, 
                  marker='s', edgecolor='black', linewidth=1.5, alpha=1.0)
    
    # 设置坐标轴和标题
    ax.set_xlabel('X (mm)', fontsize=12)
    ax.set_ylabel('Y (mm)', fontsize=12)
    ax.set_zlabel('Z (mm)', fontsize=12)
    
    title = f'Surface with {len(path_points_list)} Toolpath(s)'
    if tool_radius > 0:
        title += f' (Tool Radius: {tool_radius:.3f})'
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # 添加图例
    if len(path_points_list) <= 8:
        ax.legend(loc='upper left', bbox_to_anchor=(0, 1), fontsize=10)
    
    # 设置视角
    ax.view_init(elev=25, azim=45)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"曲面和刀具路径图已保存到: {filename}")
    
    return fig, ax

# 使用示例函数
def example_usage():
    """使用示例"""
    
    # 示例1: 基本曲面和路径
    # surface_points = your_surface_points  # shape: (N^2, 3)
    # path1 = your_path1_points             # shape: (M1, 3)
    # path2 = your_path2_points             # shape: (M2, 3)
    # paths = [path1, path2]
    
    # visualize_surface_with_paths(
    #     surface_points, paths, 
    #     grid_size=100,  # 如果是100x100的网格
    #     filename='output/surface_with_paths.png',
    #     path_colors=['red', 'blue'],
    #     path_widths=[3.0, 2.5]
    # )
    
    # 示例2: 刀具路径专用版本
    # toolpath_data = {
    #     'paths': [path1, path2, path3],
    #     'tool_radius': 0.005,  # 5mm工具半径
    #     'cutting_direction': None
    # }
    
    # visualize_surface_with_toolpaths(
    #     surface_points, toolpath_data,
    #     grid_size=100,
    #     filename='output/surface_with_toolpaths.png',
    #     show_tool_contacts=True
    # )
    
    pass