import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from utils import to_numpy

class surface_shape_evaluator:
    """
    评估两个共享相同xy坐标的曲面形状差异
    专门针对网格数据设计，假设输入的数据为高度数据，具有形状 ((sample_size+1)**2, )，并以 x 优先排列
    """
    def __init__(self, device, Ps, target_surface_heights):
        """
        初始化曲面评估器
        
        Args:
            device: 计算设备 ('cpu' 或 'cuda')
            Ps: 采样点坐标，形状为 (sample_size+1, sample_size+1, 3) 或 (N, 3)，可以是tensor或numpy数组
            target_surface_heights: 目标曲面高度数据，形状为 ((sample_size+1)**2,)，可以是tensor或numpy数组
        """
        self.device = device
        
        # 将目标高度数据转换为numpy数组
        self.target_surface_heights = to_numpy(target_surface_heights).flatten()
        Ps_numpy = to_numpy(Ps)
        
        # 处理采样点坐标 - 只考虑网格情况
        if len(Ps_numpy.shape) == 3:  # (H, W, 3)
            self.grid_shape = (Ps_numpy.shape[0], Ps_numpy.shape[1])
            self.sample_size = Ps_numpy.shape[0]
            self.Ps_xy = Ps_numpy[:, :, :2].reshape(-1, 2)  # 取出采样点的 x 和 y 分量
        else:  # (N, 3) - 假设是正方形网格
            total_points = Ps_numpy.shape[0]
            self.sample_size = int(np.sqrt(total_points))
            assert self.sample_size * self.sample_size == total_points, \
                f"点数 {total_points} 不是完全平方数，无法构成正方形网格"
            self.grid_shape = (self.sample_size, self.sample_size)
            self.Ps_xy = Ps_numpy[:, :2]
        
        # 验证数据一致性
        expected_points = self.grid_shape[0] * self.grid_shape[1]
        assert len(self.target_surface_heights) == expected_points, \
            f"目标高度数据长度 {len(self.target_surface_heights)} 与网格大小 {expected_points} 不匹配"
        assert len(self.Ps_xy) == expected_points, \
            f"坐标点数量 {len(self.Ps_xy)} 与网格大小 {expected_points} 不匹配"
        
        # 构建基础网格数据
        self._prepare_base_grid()
        
        # 初始化状态标志
        self.has_init_data = False
        self.has_final_data = False
        
        print(f"曲面评估器初始化完成:")
        print(f"  网格大小: {self.grid_shape}")
        print(f"  总点数: {expected_points}")
    
    def _prepare_base_grid(self):
        """准备基础网格数据"""
        # 重新整形为网格格式
        X = self.Ps_xy[:, 0].reshape(self.grid_shape)
        Y = self.Ps_xy[:, 1].reshape(self.grid_shape)
        Z_target = self.target_surface_heights.reshape(self.grid_shape)
        
        self.grid_data = {
            'X': X, 
            'Y': Y,
            'Z_target': Z_target
        }
        
        # 存储网格范围信息
        self.x_range = (X.min(), X.max())
        self.y_range = (Y.min(), Y.max())
        self.z_target_range = (Z_target.min(), Z_target.max())
        
        print(f"基础网格数据准备完成:")
        print(f"  X范围: [{self.x_range[0]:.4f}, {self.x_range[1]:.4f}]")
        print(f"  Y范围: [{self.y_range[0]:.4f}, {self.y_range[1]:.4f}]")
        print(f"  目标Z范围: [{self.z_target_range[0]:.4f}, {self.z_target_range[1]:.4f}]")
    
    def set_init_heights(self, init_heights):
        """
        设置初始高度数据
        
        Args:
            init_heights: 初始曲面高度数据，可以是tensor或numpy数组
        """
        init_heights_np = to_numpy(init_heights).flatten()
        
        # 验证数据长度
        expected_length = self.grid_shape[0] * self.grid_shape[1]
        assert len(init_heights_np) == expected_length, \
            f"初始高度数据长度 ({len(init_heights_np)}) 与网格大小 ({expected_length}) 不匹配"
        
        # 存储初始数据
        self.init_heights = init_heights_np
        
        # 构建初始网格数据
        Z_init = self.init_heights.reshape(self.grid_shape)
        self.grid_data['Z_init'] = Z_init
        self.grid_data['Z_diff_init'] = Z_init - self.grid_data['Z_target']
        
        self.has_init_data = True
        self.z_init_range = (Z_init.min(), Z_init.max())
        
        print(f"初始高度数据已设置:")
        print(f"  初始Z范围: [{self.z_init_range[0]:.4f}, {self.z_init_range[1]:.4f}]")
    
    def set_final_heights(self, final_heights):
        """
        设置最终高度数据
        
        Args:
            final_heights: 最终曲面高度数据，可以是tensor或numpy数组
        """
        final_heights_np = to_numpy(final_heights).flatten()
        
        # 验证数据长度
        expected_length = self.grid_shape[0] * self.grid_shape[1]
        assert len(final_heights_np) == expected_length, \
            f"最终高度数据长度 ({len(final_heights_np)}) 与网格大小 ({expected_length}) 不匹配"
        
        # 存储最终数据
        self.final_heights = final_heights_np
        
        # 构建最终网格数据
        Z_final = self.final_heights.reshape(self.grid_shape)
        self.grid_data['Z_final'] = Z_final
        self.grid_data['Z_diff_final'] = Z_final - self.grid_data['Z_target']
        
        self.has_final_data = True
        self.z_final_range = (Z_final.min(), Z_final.max())
        
        print(f"最终高度数据已设置:")
        print(f"  最终Z范围: [{self.z_final_range[0]:.4f}, {self.z_final_range[1]:.4f}]")
    
    def visualize_surfaces_simplified(self, save_path=None, alpha=0.8):
        """
        简化的曲面对比可视化方法
        左侧：目标曲面跨两行，右侧：初始/最终曲面 + 在目标曲面上着色显示MAE + 共享colorbar
        """
        if not self.has_init_data or not self.has_final_data:
            print("警告: 需要同时设置初始和最终高度数据才能进行可视化")
            return
        
        X, Y = self.grid_data['X'], self.grid_data['Y']
        Z_target = self.grid_data['Z_target']
        Z_init = self.grid_data['Z_init']
        Z_final = self.grid_data['Z_final']
        Z_diff_init = self.grid_data['Z_diff_init']
        Z_diff_final = self.grid_data['Z_diff_final']
        
        # 计算统一的颜色范围（3D曲面）
        z_min = min(Z_target.min(), Z_init.min(), Z_final.min())
        z_max = max(Z_target.max(), Z_init.max(), Z_final.max())
        
        # 计算MAE并统一尺度（用于着色）
        mae_init = np.abs(Z_diff_init)
        mae_final = np.abs(Z_diff_final)
        mae_max = max(mae_init.max(), mae_final.max())
        
        # 创建2x4布局的图形
        fig = plt.figure(figsize=(20, 10))
        fig.suptitle('Surface Comparison Summary', fontsize=16, fontweight='bold')
        
        # 左侧：目标曲面跨两行 (位置1和5)
        ax1 = fig.add_subplot(2, 3, (1, 4), projection='3d')
        surf1 = ax1.plot_surface(X, Y, Z_target, cmap='viridis', alpha=alpha, rcount=300, ccount=300,
                           linewidth=0, antialiased=True, vmin=z_min, vmax=z_max)
        ax1.set_title('Target Surface', fontsize=14, fontweight='bold')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        
        # 第一行右侧
        # 242: 初始曲面 (第一行第二列)
        ax2 = fig.add_subplot(232, projection='3d')
        surf2 = ax2.plot_surface(X, Y, Z_init, cmap='viridis', alpha=alpha, 
                           rcount=300, ccount=300, linewidth=0, antialiased=True, vmin=z_min, vmax=z_max)
        ax2.set_title('Initial Our Surface', fontsize=14, fontweight='bold')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        
        # 243: 在目标曲面上显示初始MAE (第一行第三列)
        ax3 = fig.add_subplot(233, projection='3d')
        # 将MAE值归一化用于着色
        mae_init_normalized = mae_init / mae_max  # 归一化到[0,1]
        colors_init = plt.cm.jet(mae_init_normalized)  # 使用coolwarm colormap
        
        surf3 = ax3.plot_surface(X, Y, Z_target, facecolors=colors_init, rcount=300, ccount=300,
                           alpha=alpha, linewidth=0, antialiased=True)
        ax3.set_title('Target Surface with Initial MAE Coloring', fontsize=14, fontweight='bold')
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        ax3.set_zlabel('Z')
    

        
        # 第二行右侧
        # 246: 最终曲面 (第二行第二列)
        ax6 = fig.add_subplot(235, projection='3d')
        surf6 = ax6.plot_surface(X, Y, Z_final, cmap='viridis', alpha=alpha, rcount=300, ccount=300, 
                           linewidth=0, antialiased=True, vmin=z_min, vmax=z_max)
        ax6.set_title('Final Our Surface', fontsize=14, fontweight='bold')
        ax6.set_xlabel('X')
        ax6.set_ylabel('Y')
        ax6.set_zlabel('Z')
        
        # 247: 在目标曲面上显示最终MAE (第二行第三列)
        ax7 = fig.add_subplot(236, projection='3d')
        # 将MAE值归一化用于着色
        mae_final_normalized = mae_final / mae_max  # 使用统一的最大值归一化
        colors_final = plt.cm.jet(mae_final_normalized)  # 使用coolwarm colormap
        
        surf7 = ax7.plot_surface(X, Y, Z_target, facecolors=colors_final, rcount=300, ccount=300,
                           alpha=alpha, linewidth=0, antialiased=True)
        ax7.set_title('Target Surface with Final MAE Coloring', fontsize=14, fontweight='bold')
        ax7.set_xlabel('X')
        ax7.set_ylabel('Y')
        ax7.set_zlabel('Z')
        
        # 添加两个共享colorbar
        # 左侧3D曲面的共享colorbar
        fig.colorbar(surf1, ax=[ax1, ax2, ax6], shrink=0.8, aspect=25, pad=0.05, 
                    label='Surface Height')
        
        # 右侧MAE着色曲面的共享colorbar
        # 创建用于colorbar的ScalarMappable对象
        mappable_mae = plt.cm.ScalarMappable(cmap='jet')
        mappable_mae.set_array(np.array([0, mae_max]))  # 设置范围从0到mae_max
        mappable_mae.set_clim(0, mae_max)
        
        cbar_mae = fig.colorbar(mappable_mae, ax=[ax3, ax7], shrink=0.8, aspect=25, pad=0.10)
        cbar_mae.set_label('Mean Absolute Error', rotation=270, labelpad=15)
        # 设置科学计数法格式
        cbar_mae.formatter.set_powerlimits((0, 0))
        cbar_mae.update_ticks()
        
        #plt.tight_layout()
        
        # 保存图像
        if save_path:
            base_path = os.path.dirname(save_path)
            if not os.path.exists(base_path):
                os.makedirs(base_path)
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"简化对比图已保存到: {save_path}")
        
        
        plt.close()
        
        return fig
    
    def visualize_mae_distribution(self, save_path=None):
        """
        可视化MAE分布信息
        包含：CDF曲线对比图、CDF阶梯图、改善统计信息
        """
        if not self.has_init_data or not self.has_final_data:
            print("警告: 需要同时设置初始和最终高度数据才能进行可视化")
            return
        
        # 计算MAE数据
        mae_init = np.abs(self.init_heights - self.target_surface_heights)
        mae_final = np.abs(self.final_heights - self.target_surface_heights)
        
        # 创建1x3布局的图形
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('MAE Distribution Analysis', fontsize=16, fontweight='bold')
        
        # 1. CDF曲线对比图
        ax1.set_title('Cumulative Distribution Function (CDF)', fontsize=14, fontweight='bold')
        
        # 计算累积分布
        mae_init_sorted = np.sort(mae_init)
        mae_final_sorted = np.sort(mae_final)
        percentiles_init = np.arange(1, len(mae_init_sorted) + 1) / len(mae_init_sorted) * 100
        percentiles_final = np.arange(1, len(mae_final_sorted) + 1) / len(mae_final_sorted) * 100
        
        # 绘制CDF曲线
        ax1.plot(mae_init_sorted, percentiles_init, color='lightcoral', linewidth=2, 
                 label=f'Initial (Median: {np.median(mae_init):.2e})', alpha=0.8)
        ax1.plot(mae_final_sorted, percentiles_final, color='steelblue', linewidth=2, 
                 label=f'Final (Median: {np.median(mae_final):.2e})')
        
        # 添加关键分位数标记
        percentiles_to_mark = [50, 90, 95]  # 中位数、90%、95%分位数
        for p in percentiles_to_mark:
            init_value = np.percentile(mae_init, p)
            final_value = np.percentile(mae_final, p)
            
            # 标记初始值
            ax1.axvline(init_value, color='lightcoral', linestyle=':', alpha=0.5)
            ax1.axhline(p, color='gray', linestyle=':', alpha=0.3)
            
            # 标记最终值
            ax1.axvline(final_value, color='steelblue', linestyle=':', alpha=0.5)
            
            # 添加改善箭头（如果有改善）
            if final_value < init_value:
                ax1.annotate('', xy=(final_value, p), xytext=(init_value, p),
                            arrowprops=dict(arrowstyle='<->', color='green', lw=1.5))
                # 添加改善数值标注
                improvement = init_value - final_value
                ax1.text((init_value + final_value) / 2, p + 2, f'-{improvement:.2e}',
                        ha='center', va='bottom', fontsize=8, color='green', fontweight='bold')
    
        ax1.set_xlabel('Mean Absolute Error')
        ax1.set_ylabel('Cumulative Percentage (%)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, None)
        ax1.set_ylim(0, 100)
        
        # 2. CDF阶梯图
        ax2.set_title('CDF Step Plot', fontsize=14, fontweight='bold')
        
        # 使用统一的MAE范围创建bins
        max_mae = max(mae_init.max(), mae_final.max())
        bins = np.linspace(0, max_mae, 100)
        
        # 计算每个bin的累积概率
        hist_init, _ = np.histogram(mae_init, bins=bins)
        hist_final, _ = np.histogram(mae_final, bins=bins)
        
        cdf_init = np.cumsum(hist_init) / len(mae_init) * 100
        cdf_final = np.cumsum(hist_final) / len(mae_final) * 100
        
        # 绘制阶梯图
        ax2.step(bins[1:], cdf_init, where='post', color='lightcoral', linewidth=2, 
                 label=f'Initial CDF', alpha=0.8)
        ax2.step(bins[1:], cdf_final, where='post', color='steelblue', linewidth=2, 
                 label=f'Final CDF')
        
        # 填充区域显示改善
        ax2.fill_between(bins[1:], cdf_init, cdf_final, where=(cdf_final >= cdf_init), 
                         color='green', alpha=0.3, interpolate=True, label='Improvement Area')
        
        ax2.set_xlabel('Mean Absolute Error')
        ax2.set_ylabel('Cumulative Percentage (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, max_mae)
        ax2.set_ylim(0, 100)
        
        
        #plt.tight_layout()
        
        # 保存图像
        if save_path:
            base_path = os.path.dirname(save_path)
            if not os.path.exists(base_path):
                os.makedirs(base_path)
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"MAE分布分析图已保存到: {save_path}")
        
        plt.close()
        
        return fig