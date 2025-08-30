import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import cdist, directed_hausdorff
from sklearn.neighbors import NearestNeighbors
from scipy.interpolate import griddata
import os
import config as cfg
from utils import to_numpy

class surface_shape_evaluator:
    """
    评估两个共享相同xy坐标的曲面形状差异
    假设输入的数据为高度数据，具有形状 ((sample_size+1)**2, )，并以 x 优先排列
    """
    def __init__(self, device, Ps, target_surface_heights, our_surface_heights):
        """
        初始化曲面评估器
        
        Args:
            device: 计算设备 ('cpu' 或 'cuda')
            Ps: 采样点坐标，形状为 (sample_size+1, sample_size+1, 3) 或 (N, 3)，可以是tensor或numpy数组
            target_surface_heights: 目标曲面高度数据，形状为 ((sample_size+1)**2,)，可以是tensor或numpy数组
            our_surface_heights: 我们的曲面高度数据，形状为 ((sample_size+1)**2,)，可以是tensor或numpy数组
        """
        self.device = device
        
        # 将所有输入转换为numpy数组
        self.target_surface_heights = to_numpy(target_surface_heights).flatten()
        self.our_surface_heights = to_numpy(our_surface_heights).flatten()
        Ps_numpy = to_numpy(Ps)
        
        # 处理采样点坐标
        if len(Ps_numpy.shape) == 3:  # (H, W, 3)
            self.Ps_xy = Ps_numpy[:, :, :2].reshape(-1, 2)  # 取出采样点的 x 和 y 分量
            self.sample_size = Ps_numpy.shape[0]
            self.grid_shape = (Ps_numpy.shape[0], Ps_numpy.shape[1])
        else:  # (N, 3)
            self.Ps_xy = Ps_numpy[:, :2]
            self.sample_size = cfg.sample_size+1
            self.grid_shape = (self.sample_size, self.sample_size)
    
        # 验证数据一致性
        assert len(self.target_surface_heights) == len(self.our_surface_heights), \
            "目标高度和我们的高度数据长度不匹配"
        assert len(self.target_surface_heights) == len(self.Ps_xy), \
            "高度数据与坐标点数量不匹配"
        
        # 构建完整的3D点云
        self.target_points_3d = np.column_stack([self.Ps_xy, self.target_surface_heights])
        self.our_points_3d = np.column_stack([self.Ps_xy, self.our_surface_heights])
        
        # 构建网格数据用于可视化
        self._prepare_grid_data()
    
    def _prepare_grid_data(self):
        """准备网格数据用于3D绘图"""
        try:
            # 重新整形为网格格式
            X = self.Ps_xy[:, 0].reshape(self.grid_shape)
            Y = self.Ps_xy[:, 1].reshape(self.grid_shape)
            Z_target = self.target_surface_heights.reshape(self.grid_shape)
            Z_our = self.our_surface_heights.reshape(self.grid_shape)
            
            self.grid_data = {
                'X': X, 'Y': Y,
                'Z_target': Z_target,
                'Z_our': Z_our,
                'Z_diff': Z_our - Z_target
            }
            self.has_grid_data = True
        except:
            self.has_grid_data = False
            print("警告: 无法创建规整网格，将使用散点显示")
            
    def update_our_surface(self, new_our_surface_heights):
        """
        更新我们的曲面高度数据以及相关的成员变量
        
        Args:
            new_our_surface_heights: 新的我们的曲面高度数据，形状为 ((sample_size+1)**2,)，可以是tensor或numpy数组
        """
        # 将新的高度数据转换为numpy数组
        new_heights = to_numpy(new_our_surface_heights).flatten()
        
        # 验证数据长度一致性
        assert len(new_heights) == len(self.our_surface_heights), \
            f"新高度数据长度 ({len(new_heights)}) 与原数据长度 ({len(self.our_surface_heights)}) 不匹配"
        
        # 更新我们的曲面高度数据
        self.our_surface_heights = new_heights
        
        # 更新3D点云数据
        self.our_points_3d = np.column_stack([self.Ps_xy, self.our_surface_heights])
        
        # 更新网格数据
        if self.has_grid_data:
            try:
                Z_our = self.our_surface_heights.reshape(self.grid_shape)
                self.grid_data['Z_our'] = Z_our
                self.grid_data['Z_diff'] = Z_our - self.grid_data['Z_target']
            except Exception as e:
                print(f"警告: 更新网格数据时出错: {e}")
                # 重新尝试准备网格数据
                self._prepare_grid_data()
        
        print("曲面数据已成功更新")
    
    def calculate_height_metrics(self):
        """计算基于高度差异的指标"""
        height_diff = self.our_surface_heights - self.target_surface_heights
        
        return {
            'mean_error': np.mean(height_diff),
            'mae': np.mean(np.abs(height_diff)),  # 平均绝对误差
            'rmse': np.sqrt(np.mean(height_diff**2)),  # 均方根误差
            'max_error': np.max(np.abs(height_diff)),
            'std_error': np.std(height_diff),
            'height_differences': height_diff
        }
    
    def calculate_3d_metrics(self):
        """计算3D空间中的距离指标"""
        # RMS距离
        distances = np.linalg.norm(self.our_points_3d - self.target_points_3d, axis=1)
        rms_distance = np.sqrt(np.mean(distances**2))
        
        # 使用基于网格的分块计算Hausdorff距离
        if self.has_grid_data:
            hausdorff_dist, forward_hausdorff, backward_hausdorff = self._calculate_grid_based_hausdorff()
            print(f"使用网格分块方法计算Hausdorff距离 (数据点数: {len(self.target_points_3d)})")
        else:
            # 如果没有网格数据，使用采样方法作为后备
            hausdorff_dist, forward_hausdorff, backward_hausdorff = self._calculate_sampled_hausdorff()
            print(f"使用采样方法计算Hausdorff距离 (数据点数: {len(self.target_points_3d)})")
    
        return {
            'rms_3d': rms_distance,
            'distances_3d': distances,
            'mean_3d': np.mean(distances),
            'max_3d': np.max(distances),
            'std_3d': np.std(distances),
            'hausdorff': hausdorff_dist,
            'forward_hausdorff': forward_hausdorff,
            'backward_hausdorff': backward_hausdorff
        }

    def _calculate_grid_based_hausdorff(self, block_size_ratio=0.1):
        """
        基于网格的分块Hausdorff距离计算
        
        Args:
            block_size_ratio: 分块大小占总网格大小的比例，默认0.1（即10%）
        """
        X, Y = self.grid_data['X'], self.grid_data['Y']
        Z_target = self.grid_data['Z_target']
        Z_our = self.grid_data['Z_our']
        
        grid_height, grid_width = X.shape
        
        # 计算分块大小，确保至少有4x4的块
        block_height = max(4, int(grid_height * block_size_ratio))
        block_width = max(4, int(grid_width * block_size_ratio))
        
        # 限制最大块大小以控制计算复杂度
        max_block_size = 64
        block_height = min(block_height, max_block_size)
        block_width = min(block_width, max_block_size)
        
        print(f"网格大小: {grid_height}x{grid_width}, 分块大小: {block_height}x{block_width}")
        
        forward_max_distances = []  # 从our到target的最大距离
        backward_max_distances = []  # 从target到our的最大距离
        block_count = 0
        
        # 遍历所有块
        for i in range(0, grid_height, block_height):
            for j in range(0, grid_width, block_width):
                i_end = min(i + block_height, grid_height)
                j_end = min(j + block_width, grid_width)
                
                # 提取块数据
                block_x = X[i:i_end, j:j_end].flatten()
                block_y = Y[i:i_end, j:j_end].flatten()
                block_z_target = Z_target[i:i_end, j:j_end].flatten()
                block_z_our = Z_our[i:i_end, j:j_end].flatten()
                
                # 过滤掉无效值
                valid_mask = np.isfinite(block_x) & np.isfinite(block_y) & np.isfinite(block_z_target) & np.isfinite(block_z_our)
                if not np.any(valid_mask):
                    continue
                    
                block_x = block_x[valid_mask]
                block_y = block_y[valid_mask]
                block_z_target = block_z_target[valid_mask]
                block_z_our = block_z_our[valid_mask]
                
                if len(block_x) == 0:
                    continue
                
                # 构建3D点
                block_target_3d = np.column_stack([block_x, block_y, block_z_target])
                block_our_3d = np.column_stack([block_x, block_y, block_z_our])
                
                # 计算块内的距离
                block_distances = np.linalg.norm(block_our_3d - block_target_3d, axis=1)
                
                if len(block_distances) > 0:
                    forward_max_distances.append(np.max(block_distances))
                    backward_max_distances.append(np.max(block_distances))  # 由于共享相同xy坐标，forward和backward相同
                    block_count += 1
        
        print(f"处理了 {block_count} 个有效块")
        
        if len(forward_max_distances) == 0:
            print("警告: 没有找到有效的分块数据")
            return 0.0, 0.0, 0.0
        
        # 计算最终的Hausdorff距离
        forward_hausdorff = np.max(forward_max_distances)
        backward_hausdorff = np.max(backward_max_distances)
        hausdorff_dist = max(forward_hausdorff, backward_hausdorff)
        
        return hausdorff_dist, forward_hausdorff, backward_hausdorff

    def _calculate_sampled_hausdorff(self, sample_ratio=0.01):
        """
        采样方法计算Hausdorff距离（作为网格方法的后备）
        
        Args:
            sample_ratio: 采样比例，默认1%
        """
        n_points = len(self.target_points_3d)
        sample_size = max(1000, int(n_points * sample_ratio))
        sample_size = min(sample_size, n_points)  # 不能超过总点数
        
        print(f"使用采样法: 从{n_points}个点中采样{sample_size}个点")
        
        if sample_size < n_points:
            # 随机采样
            indices = np.random.choice(n_points, sample_size, replace=False)
            sampled_our = self.our_points_3d[indices]
            sampled_target = self.target_points_3d[indices]
        else:
            # 使用全部数据
            sampled_our = self.our_points_3d
            sampled_target = self.target_points_3d
        
        # 由于共享相同xy坐标，直接计算z方向距离作为近似
        z_distances = np.abs(sampled_our[:, 2] - sampled_target[:, 2])
        hausdorff_dist = np.max(z_distances)
        
        return hausdorff_dist, hausdorff_dist, hausdorff_dist

    def _calculate_adaptive_hausdorff(self):
        """
        自适应Hausdorff距离计算：根据数据规模自动选择方法
        """
        n_points = len(self.target_points_3d)
        
        if n_points < 1000:
            # 小数据集：使用精确方法
            from scipy.spatial.distance import directed_hausdorff
            forward_hausdorff = directed_hausdorff(self.our_points_3d, self.target_points_3d)[0]
            backward_hausdorff = directed_hausdorff(self.target_points_3d, self.our_points_3d)[0]
            hausdorff_dist = max(forward_hausdorff, backward_hausdorff)
            print("使用精确方法计算Hausdorff距离")
            return hausdorff_dist, forward_hausdorff, backward_hausdorff
        
        elif self.has_grid_data:
            # 有网格数据：使用分块方法
            return self._calculate_grid_based_hausdorff()
        
        else:
            # 无网格数据：使用采样方法
            return self._calculate_sampled_hausdorff()

    def calculate_3d_metrics_fast(self, hausdorff_method='grid'):
        """
        快速版本的3D指标计算，专门针对大数据集优化
        
        Args:
            hausdorff_method: 'grid', 'sample', 'adaptive', 'skip'
        """
        # RMS距离等基本指标（这些计算很快）
        distances = np.linalg.norm(self.our_points_3d - self.target_points_3d, axis=1)
        rms_distance = np.sqrt(np.mean(distances**2))
        
        # 根据方法选择Hausdorff计算
        if hausdorff_method == 'grid':
            if self.has_grid_data:
                hausdorff_dist, forward_hausdorff, backward_hausdorff = self._calculate_grid_based_hausdorff()
            else:
                hausdorff_dist, forward_hausdorff, backward_hausdorff = self._calculate_sampled_hausdorff()
        elif hausdorff_method == 'sample':
            hausdorff_dist, forward_hausdorff, backward_hausdorff = self._calculate_sampled_hausdorff()
        elif hausdorff_method == 'adaptive':
            hausdorff_dist, forward_hausdorff, backward_hausdorff = self._calculate_adaptive_hausdorff()
        elif hausdorff_method == 'skip':
            hausdorff_dist = forward_hausdorff = backward_hausdorff = float('nan')
            print("跳过Hausdorff距离计算")
        else:
            raise ValueError(f"未知的Hausdorff计算方法: {hausdorff_method}")
        
        return {
            'rms_3d': rms_distance,
            'distances_3d': distances,
            'mean_3d': np.mean(distances),
            'max_3d': np.max(distances),
            'std_3d': np.std(distances),
            'hausdorff': hausdorff_dist,
            'forward_hausdorff': forward_hausdorff,
            'backward_hausdorff': backward_hausdorff,
            'method_used': hausdorff_method
        }

    # 修改evaluate_all_metrics以支持选择计算方法
    def evaluate_all_metrics(self, fast_mode=True, hausdorff_method='grid'):
        """
        计算所有评估指标
        
        Args:
            fast_mode: 是否使用快速模式（推荐用于大数据集）
            hausdorff_method: Hausdorff距离计算方法
        """
        height_metrics = self.calculate_height_metrics()
        
        if fast_mode:
            metrics_3d = self.calculate_3d_metrics_fast(hausdorff_method=hausdorff_method)
        else:
            metrics_3d = self.calculate_3d_metrics()
        
        #surface_stats = self.calculate_surface_statistics()
        
        return {
            'height_metrics': height_metrics,
            '3d_metrics': metrics_3d,
            #'surface_statistics': surface_stats
        }

    def print_evaluation_summary(self, fast_mode=True, hausdorff_method='grid', save_to_file=None):
        """
        打印评估结果摘要并可选择保存到文件
        
        Args:
            fast_mode: 是否使用快速模式
            hausdorff_method: Hausdorff距离计算方法
            save_to_file: 保存路径，None表示不保存
        """
        results = self.evaluate_all_metrics(fast_mode=fast_mode, hausdorff_method=hausdorff_method)
        
        # 构建输出内容
        output_lines = []
        output_lines.append("=== 曲面评估结果摘要 ===")
        output_lines.append("")
        output_lines.append("高度差异指标:")
        output_lines.append(f"  平均误差: {results['height_metrics']['mean_error']:.6f}")
        output_lines.append(f"  平均绝对误差: {results['height_metrics']['mae']:.6f}")
        output_lines.append(f"  均方根误差: {results['height_metrics']['rmse']:.6f}")
        output_lines.append(f"  最大误差: {results['height_metrics']['max_error']:.6f}")
        output_lines.append(f"  误差标准差: {results['height_metrics']['std_error']:.6f}")
        
        output_lines.append("")
        output_lines.append("3D空间距离指标:")
        output_lines.append(f"  RMS距离: {results['3d_metrics']['rms_3d']:.6f}")
        output_lines.append(f"  平均3D距离: {results['3d_metrics']['mean_3d']:.6f}")
        output_lines.append(f"  最大3D距离: {results['3d_metrics']['max_3d']:.6f}")
        
        if not np.isnan(results['3d_metrics']['hausdorff']):
            output_lines.append(f"  Hausdorff距离: {results['3d_metrics']['hausdorff']:.6f}")
            if 'method_used' in results['3d_metrics']:
                output_lines.append(f"  (使用{results['3d_metrics']['method_used']}方法)")
        else:
            output_lines.append("  Hausdorff距离: 已跳过")
        
        output_lines.append("========================")
        
        # 打印到控制台
        for line in output_lines:
            print(line)
        
        # 保存到文件
        if save_to_file:
            try:
                # 确保目录存在
                os.makedirs(os.path.dirname(save_to_file), exist_ok=True)
                
                with open(save_to_file, 'w', encoding='utf-8') as f:
                    for line in output_lines:
                        f.write(line + '\n')
                
                print(f"\n评估结果已保存到: {save_to_file}")
            except Exception as e:
                print(f"保存文件时出错: {e}")
        
        return results
    
    
    
    def visualize_surfaces_comprehensive(self, save_path=None, alpha=0.8):
        """
        全面展示曲面对比的可视化方法
        第一张图：2x2布局展示3D曲面
        第二张图：2D平面表示方法
        """
        if not self.has_grid_data:
            print("警告: 无网格数据，无法使用此可视化方法")
            return
        
        X, Y = self.grid_data['X'], self.grid_data['Y']
        Z_target = self.grid_data['Z_target']
        Z_our = self.grid_data['Z_our']
        Z_diff = self.grid_data['Z_diff']
        
        # 计算两个曲面的共同颜色范围
        z_min = min(Z_target.min(), Z_our.min())
        z_max = max(Z_target.max(), Z_our.max())
        
        # 第一张图：2x2布局的3D曲面
        fig1 = plt.figure(figsize=(16, 12))
        fig1.suptitle('3D Surface Comparison', fontsize=16, fontweight='bold')
        
        # 1. 目标曲面
        ax1 = fig1.add_subplot(221, projection='3d')
        surf1 = ax1.plot_surface(X, Y, Z_target, cmap='viridis', alpha=alpha, rcount=300, ccount=300,
                               linewidth=0, antialiased=True, vmin=z_min, vmax=z_max)
        ax1.set_title('Target Surface', fontsize=14, fontweight='bold')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        fig1.colorbar(surf1, ax=ax1, shrink=0.6)
        
        # 2. 我们的曲面
        ax2 = fig1.add_subplot(222, projection='3d')
        surf2 = ax2.plot_surface(X, Y, Z_our, cmap='viridis', alpha=alpha, rcount=300, ccount=300, 
                               linewidth=0, antialiased=True, vmin=z_min, vmax=z_max)
        ax2.set_title('Our Surface', fontsize=14, fontweight='bold')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        fig1.colorbar(surf2, ax=ax2, shrink=0.6)
        
        # 3. 叠加显示
        ax3 = fig1.add_subplot(223, projection='3d')
        surf3_target = ax3.plot_surface(X, Y, Z_target, cmap='Blues', alpha=0.6, rcount=300, ccount=300, 
                                      linewidth=0, antialiased=True)
        surf3_our = ax3.plot_surface(X, Y, Z_our, cmap='Reds', alpha=0.6, rcount=300, ccount=300,
                                   linewidth=0, antialiased=True)
        ax3.set_title('Overlay Comparison: (Reds: our surface)', fontsize=14, fontweight='bold')
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        ax3.set_zlabel('Z')
        
        # 4. 差异曲面 - 在目标曲面几何上显示高度差
        ax4 = fig1.add_subplot(224, projection='3d')
        # 计算归一化的高度差用于着色
        diff_normalized = (Z_diff - Z_diff.min()) / (Z_diff.max() - Z_diff.min())
        colors = plt.cm.RdBu_r(diff_normalized)

        surf4 = ax4.plot_surface(X, Y, Z_target, facecolors=colors, rcount=300, ccount=300,
                               alpha=alpha, linewidth=0, antialiased=True)
        ax4.set_title('Target Surface with Height Difference Coloring', fontsize=14, fontweight='bold')
        ax4.set_xlabel('X')
        ax4.set_ylabel('Y')
        ax4.set_zlabel('Z')

        # 手动创建colorbar
        mappable = plt.cm.ScalarMappable(cmap='RdBu_r')
        mappable.set_array(Z_diff)
        cbar4 = fig1.colorbar(mappable, ax=ax4, shrink=0.6)
        cbar4.set_label('Height Difference (Ours - Target)', rotation=270, labelpad=15)
        
        plt.tight_layout()
        
        # 第二张图：2D平面表示方法
        fig2 = plt.figure(figsize=(20, 10))
        fig2.suptitle('2D Representation Methods', fontsize=16, fontweight='bold')
        
        # 1. 等高线图对比
        ax5 = fig2.add_subplot(231)
        contour1 = ax5.contour(X, Y, Z_target, levels=15, colors='blue', alpha=0.8, linewidths=1.5)
        contour2 = ax5.contour(X, Y, Z_our, levels=15, colors='red', alpha=0.8, linewidths=1.5)
        ax5.set_title('Contour Comparison', fontsize=12, fontweight='bold')
        ax5.set_xlabel('X')
        ax5.set_ylabel('Y')
        ax5.legend(['Target', 'Ours'], loc='upper right')
        ax5.grid(True, alpha=0.3)
        
        # 2. 目标曲面热力图（使用统一尺度）
        ax6 = fig2.add_subplot(232)
        heatmap1 = ax6.imshow(Z_target, cmap='viridis', aspect='auto', origin='lower', 
                             extent=[X.min(), X.max(), Y.min(), Y.max()],
                             vmin=z_min, vmax=z_max)
        ax6.set_title('Target Surface Heatmap', fontsize=12, fontweight='bold')
        ax6.set_xlabel('X')
        ax6.set_ylabel('Y')
        fig2.colorbar(heatmap1, ax=ax6)
        
        # 3. 我们曲面热力图（使用统一尺度）
        ax7 = fig2.add_subplot(233)
        heatmap2 = ax7.imshow(Z_our, cmap='viridis', aspect='auto', origin='lower',
                             extent=[X.min(), X.max(), Y.min(), Y.max()],
                             vmin=z_min, vmax=z_max)
        ax7.set_title('Our Surface Heatmap', fontsize=12, fontweight='bold')
        ax7.set_xlabel('X')
        ax7.set_ylabel('Y')
        fig2.colorbar(heatmap2, ax=ax7)
        
        # 4. 差异热力图
        ax8 = fig2.add_subplot(234)
        diff_max = np.max(np.abs(Z_diff))
        heatmap3 = ax8.imshow(Z_diff, cmap='RdBu_r', aspect='auto', origin='lower',
                             extent=[X.min(), X.max(), Y.min(), Y.max()],
                             vmin=-diff_max, vmax=diff_max)
        ax8.set_title('Difference Heatmap', fontsize=12, fontweight='bold')
        ax8.set_xlabel('X')
        ax8.set_ylabel('Y')
        fig2.colorbar(heatmap3, ax=ax8)
        
        # 5. 填充等高线图（差异）
        ax9 = fig2.add_subplot(235)
        contourf1 = ax9.contourf(X, Y, Z_diff, levels=20, cmap='RdBu_r', alpha=0.8)
        ax9.set_title('Difference Filled Contours', fontsize=12, fontweight='bold')
        ax9.set_xlabel('X')
        ax9.set_ylabel('Y')
        fig2.colorbar(contourf1, ax=ax9)
        
        # 6. 误差分布直方图
        ax10 = fig2.add_subplot(236)
        height_diff = self.our_surface_heights - self.target_surface_heights
        ax10.hist(height_diff, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
        ax10.axvline(0, color='red', linestyle='--', alpha=0.7, label='Zero Error')
        ax10.axvline(np.mean(height_diff), color='orange', linestyle='--', alpha=0.7, 
                    label=f'Mean: {np.mean(height_diff):.4f}')
        ax10.set_title('Height Error Distribution', fontsize=12, fontweight='bold')
        ax10.set_xlabel('Height Difference')
        ax10.set_ylabel('Frequency')
        ax10.legend()
        ax10.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图像
        if save_path:
            base_path = os.path.dirname(save_path)
            if not os.path.exists(base_path):
                os.makedirs(base_path)
            fig1.savefig(f'{base_path}/3D_surfaces.png', dpi=300, bbox_inches='tight')
            fig2.savefig(f'{base_path}/2D_representations.png', dpi=300, bbox_inches='tight')
            print(f"图像已保存到:")
            print(f"  {base_path}/3D_surfaces.png")
            print(f"  {base_path}/2D_representations.png")
        
        #plt.show()
        
        # 打印颜色范围信息
        print(f"\n=== 颜色范围信息 ===")
        print(f"统一颜色范围: [{z_min:.4f}, {z_max:.4f}]")
        print(f"目标曲面范围: [{Z_target.min():.4f}, {Z_target.max():.4f}]")
        print(f"我们曲面范围: [{Z_our.min():.4f}, {Z_our.max():.4f}]")
        print(f"差异范围: [{Z_diff.min():.4f}, {Z_diff.max():.4f}]")
# 使用示例
if __name__ == "__main__":
    # 生成示例数据
    sample_size = 50
    x = np.linspace(0, 1, sample_size)
    y = np.linspace(0, 1, sample_size)
    X, Y = np.meshgrid(x, y)
    
    # 构建采样点坐标 (sample_size, sample_size, 3)
    Ps = np.stack([X, Y, np.zeros_like(X)], axis=-1)
    
    # 目标曲面高度
    target_heights = (0.3 * np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)).flatten()
    
    # 我们的曲面高度（带噪声）
    our_heights = target_heights + 0.05 * np.random.normal(0, 1, len(target_heights))
    
    # 创建评估器
    evaluator = surface_shape_evaluator('cpu', Ps, target_heights, our_heights)
    
    # 打印评估结果
    results = evaluator.print_evaluation_summary()
    
    # 全面可视化
    evaluator.visualize_surfaces_comprehensive(save_path='test/surface_comparison.png')