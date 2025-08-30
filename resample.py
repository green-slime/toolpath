import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def resample_toolpath_by_x(original_path, num_samples):
    """
    将单条刀具路径在X轴方向上进行等距重采样。

    Args:
        original_path (np.array): 原始刀具路径点，形状为 (k, 3)，其中k是点的数量。
        num_samples (int): 在X方向上新的采样点数量。

    Returns:
        np.array: 重采样后的新路径，形状为 (num_samples, 3)。
    """
    # 检查路径在X方向是否单调，这是interp1d的要求
    # np.diff(array) 计算数组中相邻元素的差值
    x_diff = np.diff(original_path[:, 0])
    if not (np.all(x_diff > 0) or np.all(x_diff < 0)):
        print("警告: 路径在X方向上不是单调的，插值结果可能不准确。")
        # 在此可以添加更复杂的处理逻辑，如分段处理或基于弧长插值

    # 分离原始坐标
    x_original = original_path[:, 0]
    y_original = original_path[:, 1]
    z_original = original_path[:, 2]

    # 创建插值函数。'cubic'表示三次样条插值，可以获得更平滑的路径。
    # fill_value="extrapolate" 允许在原始数据范围之外进行外插，但需谨慎使用。
    f_y = interp1d(x_original, y_original, kind='linear', fill_value="extrapolate")
    f_z = interp1d(x_original, z_original, kind='linear', fill_value="extrapolate")

    x_min = 0.0
    x_max = 1.0
    
    # 生成新的等间距X坐标
    x_new = np.linspace(x_min, x_max, num_samples)

    # 使用插值函数计算新的Y和Z坐标
    y_new = f_y(x_new)
    z_new = f_z(x_new)

    # 组合成新的路径点
    # np.stack([...], axis=1) 将多个一维数组按列堆叠成一个二维数组
    new_path = np.stack([x_new, y_new, z_new], axis=1)
    #print(new_path.shape)

    return new_path


if __name__ == "__main__":
    # --- 示例 ---

    # 1. 创建一条模拟的原始刀具路径 (X不均匀)
    # 假设这是一条从等残高法计算出的路径
    t = np.linspace(0, 2 * np.pi, 50)
    x_original_demo = t**1.5  # X坐标非线性增长
    y_original_demo = np.sin(t)
    z_original_demo = np.cos(t) * 0.5 + 1
    original_path_demo = np.stack([x_original_demo, y_original_demo, z_original_demo], axis=1)


    # 2. 对路径进行重采样
    N_new_samples = 100  # 新的采样点数量
    resampled_path = resample_toolpath_by_x(original_path_demo, N_new_samples)


    # 3. 可视化对比
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制原始路径
    ax.plot(original_path_demo[:, 0], original_path_demo[:, 1], original_path_demo[:, 2], 
            'o-', label='Original Path', color='blue', markersize=4)
    # 绘制重采样后的路径
    ax.plot(resampled_path[:, 0], resampled_path[:, 1], resampled_path[:, 2], 
            'x-', label=f'Resampled Path (N={N_new_samples})', color='red', markersize=5, linewidth=1)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.title("Toolpath Resampling in X-direction")
    plt.savefig("resampled_toolpath.png")
