import torch
import numpy as np

def cox_de_boor_basis(t, i, p, knots):
    """
    Cox-de Boor递归算法计算B样条基函数
    t: 参数值
    i: 基函数索引
    p: B样条次数
    knots: 节点向量
    """
    if p == 0:
        return 1.0 if knots[i] <= t < knots[i+1] else 0.0
    
    # 递归计算
    left_coeff = 0.0
    right_coeff = 0.0
    
    # 左边项
    if knots[i+p] != knots[i]:
        left_coeff = (t - knots[i]) / (knots[i+p] - knots[i])
    
    # 右边项
    if knots[i+p+1] != knots[i+1]:
        right_coeff = (knots[i+p+1] - t) / (knots[i+p+1] - knots[i+1])
    
    return (left_coeff * cox_de_boor_basis(t, i, p-1, knots) + 
            right_coeff * cox_de_boor_basis(t, i+1, p-1, knots))

def create_uniform_knots(n_control, degree):
    """
    创建均匀节点向量
    n_control: 控制点数量
    degree: B样条次数
    """
    n_knots = n_control + degree + 1
    knots = np.zeros(n_knots)
    
    # 均匀节点向量：前degree+1个为0，后degree+1个为1，中间均匀分布
    for i in range(n_knots):
        if i <= degree:
            knots[i] = 0.0
        elif i >= n_control:
            knots[i] = 1.0
        else:
            knots[i] = (i - degree) / (n_control - degree)
    
    return knots

def compute_basis_matrix(t_samples, n_control, degree):
    """
    计算B样条基函数矩阵
    t_samples: [path_len] 参数采样点
    n_control: 控制点数量
    degree: B样条次数
    
    Returns: [path_len, n_control] 基函数矩阵
    """
    knots = create_uniform_knots(n_control, degree)
    path_len = len(t_samples)
    basis_matrix = np.zeros((path_len, n_control))
    
    for i, t in enumerate(t_samples):
        # 处理边界情况
        if t >= 1.0:
            t = 1.0 - 1e-10
        
        for j in range(n_control):
            basis_matrix[i, j] = cox_de_boor_basis(t, j, degree, knots)
    
    return basis_matrix

class BSplinePathOptimizer:
    def __init__(self, path_num, n_control, path_len, device, degree=3):
        """
        B样条路径优化器
        
        Args:
            path_num: 路径数量
            n_control: 每条路径的控制点数量
            path_len: 每条路径的采样点数量
            device: 计算设备
            degree: B样条次数
        """
        self.path_num = path_num
        self.n_control = n_control
        self.path_len = path_len
        self.device = device
        self.degree = degree
        
        # 参数采样点
        self.t_samples = torch.linspace(0, 1, path_len, device=device)
        
        # 预计算基函数矩阵
        self._precompute_basis()
        
        # 控制点 [path_num, n_control, 3] - 最后初始化
        self.control_points = self._create_control_points()
    
    def _create_control_points(self):
        """创建控制点张量（确保是叶子节点）"""
        control_points = torch.randn(self.path_num, self.n_control, 3, 
                                   dtype=torch.float32, device=self.device) * 0.1
        control_points.requires_grad_(True)
        return control_points
    
    def _precompute_basis(self):
        """预计算基函数矩阵"""
        t_numpy = self.t_samples.cpu().numpy()
        basis_numpy = compute_basis_matrix(t_numpy, self.n_control, self.degree)
        self.basis_matrix = torch.tensor(basis_numpy, dtype=torch.float32, device=self.device)
        
        #print(f"基函数矩阵形状: {self.basis_matrix.shape}")
        #print(f"基函数矩阵和（每行应该接近1）: {torch.sum(self.basis_matrix, dim=1)[:5]}")
    
    def evaluate_paths(self):
        """
        从控制点生成路径点
        Returns: [path_num, path_len, 3]
        """
        # 使用einsum进行更高效的批量矩阵乘法
        path_points = torch.einsum('tc,pcd->ptd', self.basis_matrix, self.control_points)
        return path_points
    
    def fit_to_initial_paths(self, init_path_points):
        """
        将初始路径拟合为控制点
        init_path_points: [path_num, path_len, 3]
        """
        with torch.no_grad():
            # 计算基函数矩阵的伪逆
            basis_pinv = torch.linalg.pinv(self.basis_matrix)  # [n_control, path_len]
            
            #print(f"伪逆矩阵形状: {basis_pinv.shape}")
            
            # 使用einsum进行批量拟合
            fitted_control = torch.einsum('ct,ptd->pcd', basis_pinv, init_path_points)
            
            # 重新创建控制点张量（确保是叶子节点）
            self.control_points = fitted_control.detach().clone().requires_grad_(True)
            
            #print(f"控制点范围: {torch.min(self.control_points)} ~ {torch.max(self.control_points)}")
            #print(f"控制点是叶子节点: {self.control_points.is_leaf}")
    
    def reset_control_points(self, new_values=None):
        """重设控制点（确保梯度可计算）"""
        if new_values is not None:
            self.control_points = new_values.detach().clone().requires_grad_(True)
        else:
            self.control_points = self._create_control_points()
    
    def get_parameters(self):
        """返回优化参数"""
        return [self.control_points]

def test_bspline_implementation():
    """测试B样条实现的正确性"""
    print("=== B样条实现测试 ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 测试参数
    path_num = 3
    n_control = 8
    path_len = 50
    
    # 创建B样条优化器
    optimizer = BSplinePathOptimizer(path_num, n_control, path_len, device)
    
    # 测试1：验证基函数矩阵
    print("\n1. 基函数矩阵验证:")
    basis_sum = torch.sum(optimizer.basis_matrix, dim=1)
    print(f"每行和的范围: {torch.min(basis_sum):.6f} ~ {torch.max(basis_sum):.6f}")
    print(f"理想值应该是1.0，偏差: {torch.max(torch.abs(basis_sum - 1.0)):.6f}")
    
    # 测试2：创建简单的测试路径
    print("\n2. 路径拟合测试:")
    
    # 创建简单的直线路径作为测试
    test_paths = torch.zeros(path_num, path_len, 3, device=device)
    for i in range(path_num):
        t = torch.linspace(0, 1, path_len, device=device)
        test_paths[i, :, 0] = t * (i + 1)  # x方向
        test_paths[i, :, 1] = t * 0.5      # y方向  
        test_paths[i, :, 2] = torch.sin(t * 3.14159 * 2) * 0.1  # z方向（正弦波）
    
    print(f"原始路径形状: {test_paths.shape}")
    print(f"原始路径范围: {torch.min(test_paths)} ~ {torch.max(test_paths)}")
    
    # 拟合到控制点
    optimizer.fit_to_initial_paths(test_paths)
    
    # 重新生成路径
    reconstructed_paths = optimizer.evaluate_paths()
    
    # 计算拟合误差
    fitting_error = torch.mean((reconstructed_paths - test_paths) ** 2)
    max_error = torch.max(torch.abs(reconstructed_paths - test_paths))
    
    print(f"重构路径形状: {reconstructed_paths.shape}")
    print(f"平均拟合误差: {fitting_error:.6f}")
    print(f"最大拟合误差: {max_error:.6f}")
    
    # 测试3：梯度计算
    print("\n3. 梯度计算测试:")
    
    # 确保控制点需要梯度
    print(f"控制点requires_grad: {optimizer.control_points.requires_grad}")
    
    # 重新获取路径（确保计算图连接）
    current_paths = optimizer.evaluate_paths()
    
    # 创建一个简单的损失函数
    target = torch.randn_like(current_paths)
    loss = torch.mean((current_paths - target) ** 2)
    
    print(f"损失值: {loss.item():.6f}")
    print(f"损失requires_grad: {loss.requires_grad}")
    
    # 清零之前的梯度
    if optimizer.control_points.grad is not None:
        optimizer.control_points.grad.zero_()
    
    # 计算梯度
    loss.backward()
    
    if optimizer.control_points.grad is not None:
        print(f"控制点梯度形状: {optimizer.control_points.grad.shape}")
        print(f"控制点梯度范围: {torch.min(optimizer.control_points.grad):.6f} ~ {torch.max(optimizer.control_points.grad):.6f}")
        print(f"梯度非零元素数量: {torch.sum(optimizer.control_points.grad != 0).item()}")
    else:
        print("警告：控制点梯度为None！")
        
        # 调试：检查计算图
        print("调试信息：")
        print(f"control_points.requires_grad: {optimizer.control_points.requires_grad}")
        print(f"current_paths.requires_grad: {current_paths.requires_grad}")
        print(f"loss.requires_grad: {loss.requires_grad}")
    
    # 测试4：B样条的光滑性
    print("\n4. 光滑性测试:")
    
    with torch.no_grad():
        # 计算一阶导数（速度）
        first_deriv = reconstructed_paths[:, 1:, :] - reconstructed_paths[:, :-1, :]
        
        # 计算二阶导数（加速度）
        second_deriv = first_deriv[:, 1:, :] - first_deriv[:, :-1, :]
        
        avg_velocity = torch.mean(torch.norm(first_deriv, dim=2))
        avg_acceleration = torch.mean(torch.norm(second_deriv, dim=2))
        
        print(f"平均速度模长: {avg_velocity:.6f}")
        print(f"平均加速度模长: {avg_acceleration:.6f}")
        if avg_acceleration > 1e-8:
            print(f"光滑性比率 (速度/加速度): {avg_velocity/avg_acceleration:.2f}")
        else:
            print("路径非常光滑（加速度接近0）")
    
    return optimizer

def test_gradient_flow():
    """专门测试梯度流的函数"""
    print("\n=== 专门梯度流测试 ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 简单参数
    path_num, n_control, path_len = 2, 5, 20
    
    optimizer = BSplinePathOptimizer(path_num, n_control, path_len, device)
    
    print(f"初始控制点requires_grad: {optimizer.control_points.requires_grad}")
    print(f"初始控制点是叶子节点: {optimizer.control_points.is_leaf}")
    
    # 生成路径
    paths = optimizer.evaluate_paths()
    print(f"生成路径requires_grad: {paths.requires_grad}")
    print(f"生成路径是叶子节点: {paths.is_leaf}")
    
    # 简单损失：路径点的平方和
    loss = torch.sum(paths ** 2)
    print(f"损失requires_grad: {loss.requires_grad}")
    print(f"损失值: {loss.item():.6f}")
    
    # 反向传播
    print("开始反向传播...")
    loss.backward()
    
    print(f"反向传播后，控制点是叶子节点: {optimizer.control_points.is_leaf}")
    
    if optimizer.control_points.grad is not None:
        grad_norm = torch.norm(optimizer.control_points.grad)
        grad_max = torch.max(torch.abs(optimizer.control_points.grad))
        grad_nonzero = torch.sum(optimizer.control_points.grad != 0).item()
        
        print(f"梯度计算成功！")
        print(f"梯度范数: {grad_norm:.6f}")
        print(f"梯度最大值: {grad_max:.6f}")
        print(f"非零梯度元素: {grad_nonzero}/{optimizer.control_points.numel()}")
        return True
    else:
        print("梯度计算失败！")
        print(f"control_points.grad: {optimizer.control_points.grad}")
        return False

def test_gradient_after_fitting():
    """测试拟合后的梯度计算"""
    print("\n=== 拟合后梯度测试 ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 参数
    path_num, n_control, path_len = 2, 6, 30
    
    optimizer = BSplinePathOptimizer(path_num, n_control, path_len, device)
    
    # 创建测试路径
    test_paths = torch.zeros(path_num, path_len, 3, device=device)
    for i in range(path_num):
        t = torch.linspace(0, 1, path_len, device=device)
        test_paths[i, :, 0] = t * (i + 1)
        test_paths[i, :, 1] = t * 0.5
        test_paths[i, :, 2] = torch.sin(t * 3.14159 * 2) * 0.1
    
    print("拟合前:")
    print(f"控制点是叶子节点: {optimizer.control_points.is_leaf}")
    print(f"控制点requires_grad: {optimizer.control_points.requires_grad}")
    
    # 拟合到控制点
    optimizer.fit_to_initial_paths(test_paths)
    
    print("拟合后:")
    print(f"控制点是叶子节点: {optimizer.control_points.is_leaf}")
    print(f"控制点requires_grad: {optimizer.control_points.requires_grad}")
    
    # 测试梯度
    paths = optimizer.evaluate_paths()
    loss = torch.sum(paths ** 2)
    
    print(f"损失值: {loss.item():.6f}")
    
    loss.backward()
    
    if optimizer.control_points.grad is not None:
        grad_norm = torch.norm(optimizer.control_points.grad)
        print(f"拟合后梯度计算成功！梯度范数: {grad_norm:.6f}")
        
        # 验证拟合质量
        reconstructed = optimizer.evaluate_paths()
        fitting_error = torch.mean((reconstructed - test_paths) ** 2)
        print(f"拟合误差: {fitting_error:.6f}")
        
        return True
    else:
        print("拟合后梯度计算失败！")
        return False

def test_full_optimization_cycle():
    """测试完整的优化循环"""
    print("\n=== 完整优化循环测试 ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 参数
    path_num, n_control, path_len = 2, 8, 50
    
    optimizer = BSplinePathOptimizer(path_num, n_control, path_len, device)
    
    # 创建目标路径
    target_paths = torch.zeros(path_num, path_len, 3, device=device)
    for i in range(path_num):
        t = torch.linspace(0, 1, path_len, device=device)
        target_paths[i, :, 0] = t * (i + 1)
        target_paths[i, :, 1] = torch.sin(t * 3.14159 * 3) * 0.2
        target_paths[i, :, 2] = torch.cos(t * 3.14159 * 2) * 0.1
    
    # 创建优化器
    torch_optimizer = torch.optim.Adam(optimizer.get_parameters(), lr=0.01)
    
    print("开始优化循环...")
    
    for epoch in range(10):
        torch_optimizer.zero_grad()
        
        # 生成当前路径
        current_paths = optimizer.evaluate_paths()
        
        # 计算损失
        loss = torch.mean((current_paths - target_paths) ** 2)
        
        # 反向传播
        loss.backward()
        
        # 检查梯度
        if optimizer.control_points.grad is not None:
            grad_norm = torch.norm(optimizer.control_points.grad)
        else:
            grad_norm = 0.0
            print(f"警告：第{epoch}轮梯度为None")
        
        # 更新参数
        torch_optimizer.step()
        
        if epoch % 2 == 0:
            print(f"轮次 {epoch}: 损失 = {loss.item():.6f}, 梯度范数 = {grad_norm:.6f}")
    
    final_paths = optimizer.evaluate_paths()
    final_error = torch.mean((final_paths - target_paths) ** 2)
    print(f"最终拟合误差: {final_error:.6f}")
    
    return final_error < 0.01  # 如果误差小于0.01认为成功

def test_with_real_data():
    """使用真实数据测试"""
    print("\n=== 真实数据测试 ===")
    
    try:
        import config as cfg
        import utils
        from init_with_camlib import init_with_camlib
        
        device = utils.cuda_init(0)
        
        # 加载真实数据
        Ps, init_path_points = init_with_camlib(device, use_x=True)
        
        print(f"真实数据形状: {init_path_points.shape}")
        print(f"真实数据范围: {torch.min(init_path_points)} ~ {torch.max(init_path_points)}")
        
        # 创建B样条优化器
        path_num, path_len, _ = init_path_points.shape
        n_control = path_len // 6  # 使用1/6的控制点
        
        optimizer = BSplinePathOptimizer(path_num, n_control, path_len, device)
        
        # 拟合真实数据
        optimizer.fit_to_initial_paths(init_path_points)
        
        # 重构路径
        reconstructed_paths = optimizer.evaluate_paths()
        
        # 计算拟合质量
        fitting_error = torch.mean((reconstructed_paths - init_path_points) ** 2)
        max_error = torch.max(torch.abs(reconstructed_paths - init_path_points))
        
        print(f"控制点数量: {n_control} (压缩比: {n_control/path_len:.3f})")
        print(f"拟合误差: {fitting_error:.6f}")
        print(f"最大误差: {max_error:.6f}")
        
        # 计算光滑性
        original_smoothness = compute_path_smoothness(init_path_points)
        bspline_smoothness = compute_path_smoothness(reconstructed_paths)
        
        print(f"原始路径光滑性: {original_smoothness:.6f}")
        print(f"B样条路径光滑性: {bspline_smoothness:.6f}")
        print(f"光滑性改善: {original_smoothness/bspline_smoothness:.2f}x")
        
        return optimizer, init_path_points, reconstructed_paths
        
    except ImportError as e:
        print(f"无法导入模块: {e}")
        return None

def compute_path_smoothness(path_points):
    """计算路径光滑性（二阶导数的平均模长）"""
    first_deriv = path_points[:, 1:, :] - path_points[:, :-1, :]
    second_deriv = first_deriv[:, 1:, :] - first_deriv[:, :-1, :]
    return torch.mean(torch.norm(second_deriv, dim=2))

if __name__ == "__main__":
    print("开始B样条测试...")
    
    # 测试1：基础梯度流
    print("1. 基础梯度流测试")
    gradient_ok = test_gradient_flow()
    
    # 测试2：拟合后梯度计算
    print("\n2. 拟合后梯度测试") 
    fitting_gradient_ok = test_gradient_after_fitting()
    
    # 测试3：完整优化循环
    print("\n3. 完整优化循环测试")
    optimization_ok = test_full_optimization_cycle()
    
    # 汇总结果
    print("\n=== 测试结果汇总 ===")
    print(f"基础梯度流: {'✓' if gradient_ok else '✗'}")
    print(f"拟合后梯度: {'✓' if fitting_gradient_ok else '✗'}")
    print(f"完整优化循环: {'✓' if optimization_ok else '✗'}")
    
    if all([gradient_ok, fitting_gradient_ok, optimization_ok]):
        print("\n🎉 所有测试通过！B样条实现可以正常使用。")
        
        # 如果基础测试通过，运行完整测试
        print("\n运行完整功能测试...")
        try:
            optimizer = test_bspline_implementation()
            real_test_result = test_with_real_data()
        except Exception as e:
            print(f"完整测试遇到错误: {e}")
    else:
        print("\n❌ 部分测试失败，请检查实现。")
    
    print("\n=== 测试完成 ===")