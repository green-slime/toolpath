"""
此文件计算B样条曲面的高度场、法向量等信息
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class NURBS:
    @staticmethod
    def generate_uniform_knots(num_control_points, degree):
        """
        生成均匀的节点向量
        参数:
            num_control_points: 控制点数量 n
            degree: B样条次数 p
        返回:
            均匀节点向量，长度为 n+p+1
        """
        # 内部节点数量 = n-p-1
        num_internal_knots = num_control_points - degree - 1
        
        if num_internal_knots < 0:
            raise ValueError(f"控制点数量({num_control_points})必须大于阶数({degree+1})")
        
        # 生成内部节点，数量为 n-p-1 个
        if num_internal_knots > 0:
            internal_knots = np.linspace(0, 1, num_internal_knots + 2)[1:-1]
        else:
            internal_knots = np.array([], dtype=np.float32)
        
        # 构建完整节点向量：首尾各重复degree+1次，总长度为n+p+1
        knots = np.concatenate([
            np.zeros(degree + 1),  # 起点重复p+1次
            internal_knots,        # 内部节点n-p-1个
            np.ones(degree + 1)    # 终点重复p+1次
        ])
        #print(knots)
        return knots.astype(np.float32)

    def __init__(self, control_points, degree_u=3, degree_v=3, sample_points=None, flag_large_sample_size=False):
        """
        初始化B样条曲面
        参数:
            control_points: 控制点高度场序列 [nu, nv]
            degree_u: u方向次数,默认3次
            degree_v: v方向次数,默认3次
            sample_points: 预定义采样点 [N, 2], 如果为None则不预计算
        """
        self.control_points = control_points.requires_grad_(True)
        self.degree_u = degree_u
        self.degree_v = degree_v
        
        # 自动生成节点向量
        nu, nv = control_points.shape
        with torch.no_grad():
            self.knots_u = torch.tensor(
                self.generate_uniform_knots(nu, degree_u), 
                dtype=torch.float32
            )
            self.knots_v = torch.tensor(
                self.generate_uniform_knots(nv, degree_v), 
                dtype=torch.float32
            )
        
            # 预计算基函数和导数
            if sample_points is not None and not flag_large_sample_size:
                self.cached_sample_points = sample_points
                u = sample_points[:, 0]
                v = sample_points[:, 1]
                
                self.bu = basis_u = self.basis_function_vector(u, self.degree_u, self.knots_u)
                self.bv = basis_v = self.basis_function_vector(v, self.degree_v, self.knots_v)
                self.bdu = basis_u_derivative = self.basis_function_derivative(u, self.degree_u, self.knots_u)
                self.bdv = basis_v_derivative = self.basis_function_derivative(v, self.degree_v, self.knots_v)
                
                # 创建网格: 这样的预计算可能会造成内存不足！
                # bu[N, nu, 1] * bv[N, 1, nv] = basis_grid[N, nu, nv]
                """ bu = basis_u.unsqueeze(2)
                bv = basis_v.unsqueeze(1)
                bdu = basis_u_derivative.unsqueeze(2)
                bdv = basis_v_derivative.unsqueeze(1)
                self.cached_uv_grid_weight = bu * bv
                self.cached_duv_grid_weight = bdu * bv
                self.cached_udv_grid_weight = bu * bdv """

                """memory_size = self.basis_grid.element_size() * self.basis_grid.nelement()
                print(f'shape: {self.basis_grid.shape}')
                print(f'memory:{memory_size / (1024 * 1024):.2f} MB')
                exit() """

            else:
                self.cached_sample_points = None

    def basis_function_vector(self, u, p, knots):
        """
        向量化计算B样条基函数
        参数:
            u: 参数值张量 [N]
            p: 次数
            knots: 节点向量
        返回:
            基函数值张量 [N, num_control_points]
        """
        with torch.no_grad():
            N = u.shape[0]
            n = knots.shape[0]-1
            basis = torch.zeros((N, n), dtype=torch.float32, device=u.device)
            
            # p=0时的基函数
            for i in range(n):
                # u == 1.0 时，需要特殊处理
                mask = ((u >= knots[i]) & (u < knots[i+1])) | ((torch.isclose(u,torch.tensor(1.0, dtype=torch.float32)) & (u >knots[i]) & (u <= knots[i+1])))
                basis[:, i] = mask
            #print(basis[0,:])
            def get_wrongPlace(A,B):
                # 找到不相等的元素
                not_equal = ~torch.isclose(A, B, rtol=1e-05, atol=1e-08)

                # 获取不相等元素的索引
                indices = torch.where(not_equal)
                print(indices)  # 输出: (tensor([2]),)
            # 递推计算高阶基函数
            for deg in range(1, p+1):
                for i in range(n-deg):
                    # 左项系数
                    c1 = torch.zeros_like(u)
                    d = knots[i+deg] - knots[i]
                    mask = d != 0
                    c1[mask] = (u[mask] - knots[i]) / d
                    # 右项系数
                    c2 = torch.zeros_like(u) 
                    d = knots[i+deg+1] - knots[i+1]
                    mask = d != 0
                    c2[mask] = (knots[i+deg+1] - u[mask]) / d
                    # 类似动态规划的覆写
                    basis[:, i] = c1 * basis[:, i] + c2 * basis[:, i+1]   
                    #print(basis[29,:]) 
                """ 
                # 检查基函数是否正确, deg=3, knots=[0,0,0,0,1,1,1,1]
                # 此时基函数为二项式展开
                if deg == 1:
                    print("check deg 1")
                    print(torch.allclose(basis[:,2],(1-u)))
                    print(torch.allclose(basis[:,3],u))
                    get_wrongPlace(basis[:,3],u)
                if deg == 2:
                    print("check deg 2")
                    print(torch.allclose(basis[:,1],(1-u)**2))
                    print(torch.allclose(basis[:,2],2*u*(1-u)))
                    print(torch.allclose(basis[:,3],u**2))
                if deg == 3:
                    print("check deg 3")
                    print(torch.allclose(basis[:, 0], (1-u)**3))
                    print(torch.allclose(basis[:, 1], 3*u*(1-u)**2))
                    print(torch.allclose(basis[:, 2], 3*u**2*(1-u)))
                    print(torch.allclose(basis[:, 3], u**3))
                """ 
            return basis[:, 0:n-deg]
        
    def basis_function_derivative(self, u, p, knots):
        """
        计算B样条基函数的导数
        参数:
            u: 参数值张量 [N]
            p: 次数
            knots: 节点向量
        返回:
            基函数导数值张量 [N, num_control_points]
        """
        with torch.no_grad():
            N = u.shape[0]
            n = len(knots) - p - 1  # 控制点数量
            
            if p == 0:
                return torch.zeros((N, n), dtype=torch.float32, device=u.device)
            
            # 计算p-1次基函数
            basis_p_minus_1 = self.basis_function_vector(u, p-1, knots)
            
            # 计算导数
            derivative = torch.zeros((N, n), dtype=torch.float32, device=u.device)
            for i in range(n):
                # 左项系数
                c1 = p / (knots[i+p] - knots[i]) if knots[i+p] != knots[i] else 0
                # 右项系数
                c2 = p / (knots[i+p+1] - knots[i+1]) if knots[i+p+1] != knots[i+1] else 0
                
                # 计算导数
                if i > 0:
                    derivative[:, i] += c1 * basis_p_minus_1[:, i]
                if i < n - 1:  # 修改：移除-1
                    derivative[:, i] -= c2 * basis_p_minus_1[:, i + 1]
                    
            """ # 添加检查代码
            if p == 3 and torch.allclose(knots, torch.tensor([0.,0.,0.,0.,1.,1.,1.,1.])):
                # 3次伯恩斯坦基函数的导数
                print("检查3次B样条导数:")
                print("第0个基函数导数:", torch.allclose(derivative[:, 0], -3*(1-u)**2))
                print("第1个基函数导数:", torch.allclose(derivative[:, 1], 3*(1-u)**2 - 6*u*(1-u)))
                print("第2个基函数导数:", torch.allclose(derivative[:, 2], 6*u*(1-u) - 3*u**2))
                print("第3个基函数导数:", torch.allclose(derivative[:, 3], 3*u**2))
                
                # 如果需要查看具体的差异
                for i in range(4):
                    expected = None
                    if i == 0:
                        expected = -3*(1-u)**2
                    elif i == 1:
                        expected = 3*(1-u)**2 - 6*u*(1-u)
                    elif i == 2:
                        expected = 6*u*(1-u) - 3*u**2
                    elif i == 3:
                        expected = 3*u**2
                        
                    if not torch.allclose(derivative[:, i], expected):
                        print(f"基函数{i}的导数不匹配:")
                        print("计算值:", derivative[0:5, i])
                        print("期望值:", expected[0:5]) """
            
            return derivative
        
    def basis_function_second_derivative(self, u, p, knots):
        """
        计算B样条基函数的二阶导数
        参数:
            u: 参数值张量 [N]
            p: 次数
            knots: 节点向量
        返回:
            基函数二阶导数值张量 [N, num_control_points]
        """
        # with torch.no_grad(): # 如果在训练中需要梯度，可以移除此行
        N = u.shape[0]
        n = len(knots) - p - 1  # 控制点数量
        
        # 次数小于2的基函数，其二阶导数必为0
        if p < 2:
            return torch.zeros((N, n), dtype=torch.float32, device=u.device)
        
        # 核心：计算 p-1 次基函数的一阶导数
        # 我们调用您已经实现的函数，但次数是 p-1
        # 节点向量也需要相应地缩短，但由于 basis_function_derivative 内部会自己计算n，
        # 传入完整的节点向量也是安全的，它只会计算有效的 p-1 次基函数导数。
        derivative_p_minus_1 = self.basis_function_derivative(u, p - 1, knots)
        
        # 初始化二阶导数张量
        second_derivative = torch.zeros((N, n), dtype=torch.float32, device=u.device)
        
        # 遍历所有 p 次基函数
        for i in range(n):
            # 左项系数 (与一阶导数公式中的系数完全相同)
            c1 = p / (knots[i+p] - knots[i]) if knots[i+p] != knots[i] else 0
            # 右项系数
            c2 = p / (knots[i+p+1] - knots[i+1]) if knots[i+p+1] != knots[i+1] else 0
            
            # 根据公式组合 p-1 次基函数的一阶导数
            # 注意：derivative_p_minus_1 的维度是 [N, n+1] (p-1次基函数比p次多一个)
            # 所以这里的索引是安全的
            if c1 != 0:
                second_derivative[:, i] += c1 * derivative_p_minus_1[:, i]
            if c2 != 0:
                # 注意这里是 i+1
                second_derivative[:, i] -= c2 * derivative_p_minus_1[:, i + 1]
                
        return second_derivative
        
    @staticmethod
    def checkSparsity(matrix):
        nonzero_count = torch.count_nonzero(matrix).item()
        total_elements = matrix.numel()  # 总元素数
        sparsity = 1 - nonzero_count / total_elements  # 稀疏度（零元素占比）

        #print(f"非零元素数: {nonzero_count}")
        #print(f"总元素数: {total_elements}")
        #print(f"稀疏度: {sparsity:.2%}")

        # 判断是否稀疏（示例阈值：稀疏度 > 90%）
        if sparsity > 0.9:
            return True
        else:
            return False
        
    @staticmethod
    def returnSparse(matrix):
        if(NURBS.checkSparsity(matrix)):
            return matrix.to_sparse()
        else:
            return matrix
    
    @staticmethod
    def einsum_chunked(nu, nv, uv, chunk_size=40000):
        N = nu.shape[0]
        result = torch.zeros(N, device=nu.device)
        
        for i in range(0, N, chunk_size):
            end = min(i + chunk_size, N)
            nu_chunk = nu[i:end]  # (chunk_size, U)
            nv_chunk = nv[i:end]  # (chunk_size, V)
            # 计算小块结果
            temp = nu_chunk @ uv  # (chunk_size, U) @ (U, V) -> (chunk_size, V)
            result[i:end] = (temp * nv_chunk).sum(dim=1)  # 点乘并求和
        return result   
    
    def evaluate_batch(self, sample_points, control_points, wij, batch_size=40000):
        """
        批量计算曲面上点的位置和法向量
        Args:
            sample_points: 采样点坐标 [N, 2]
            control_points: 控制点坐标 [nu, nv]
            wij: 权重系数 [nu, nv]
        Returns:
            positions: 曲面上点的位置 [N]
            normals: 曲面上点的法向量 [N, 3]
        """
        with torch.no_grad():
            N = sample_points.shape[0]
            u = sample_points[:, 0]
            v = sample_points[:, 1]
            device = control_points.device
            # 初始化结果张量
            positions = torch.zeros(N, device=device)
            normals = torch.zeros((N,3), device=device)
            weighted_control = wij * control_points  # [nu, nv]

            for start in range(0, N, batch_size):
                end = min(start + batch_size, N)
                
                current_bu = self.basis_function_vector(u[start:end], self.degree_u, self.knots_u)
                current_bv = self.basis_function_vector(v[start:end], self.degree_v, self.knots_v)
                current_bdu = self.basis_function_derivative(u[start:end], self.degree_u, self.knots_u)
                current_bdv = self.basis_function_derivative(v[start:end], self.degree_v, self.knots_v)
                
                # 计算当前批的基函数网格 (不保存完整网格)
                uv_grid = current_bu[:, :, None] * current_bv[:, None, :]  # [batch, nu, nv]
                duv_grid_weight = current_bdu[:, :, None] * current_bv[:, None, :]  # [batch, nu, nv]
                udv_grid_weight = current_bu[:, :, None] * current_bdv[:, None, :]  # [batch, nu, nv]
                
                # 计算当前批的分子贡献
                num_batch = torch.einsum('ijk,jk->i', uv_grid, weighted_control)  # [batch]  
                # 计算当前批的分母贡献
                den_batch = torch.einsum('ijk,jk->i', uv_grid, wij)  # [batch]
            
                # 计算法向量
                du = (torch.einsum('ijk,jk->i', duv_grid_weight, weighted_control)*den_batch- torch.einsum('ijk,jk->i', duv_grid_weight, wij)*num_batch)/den_batch**2  # [batch]
                dv = (torch.einsum('ijk,jk->i', udv_grid_weight, weighted_control)*den_batch- torch.einsum('ijk,jk->i', udv_grid_weight, wij)*num_batch)/den_batch**2  # [batch]

                # 计算法向量
                du = torch.stack([torch.ones_like(du), torch.zeros_like(du), du], dim=1)  # [batch,3]
                dv = torch.stack([torch.zeros_like(dv), torch.ones_like(dv), dv], dim=1)  # [batch,3]

                normal = torch.cross(du, dv, dim=1)  # [batch,3]
                normal = normal / torch.norm(normal, dim=1, keepdim=True)

                positions[start:end] = num_batch / den_batch  
                normals[start:end] = normal

            return positions, normals

    def evaluate_curvature_batch(self, sample_points, control_points, wij, batch_size=40000):
        """
        批量计算曲面上点的位置和法向量
        Args:
            sample_points: 采样点坐标 [N, 2]
            control_points: 控制点坐标 [nu, nv]
            wij: 权重系数 [nu, nv]
        Returns:
            positions: 曲面上点的位置 [N]
            normals: 曲面上点的法向量 [N, 3]
            k1, k2: 主曲率 [N]
            J: 雅可比矩阵 [N, 3, 2]
        """
        with torch.no_grad():
            # 因为现在要分行计算，预计算意义不大
            u = sample_points[:, 0]
            v = sample_points[:, 1]    
            self.bu = self.basis_function_vector(u, self.degree_u, self.knots_u)
            self.bv = self.basis_function_vector(v, self.degree_v, self.knots_v)
            self.bdu = self.basis_function_derivative(u, self.degree_u, self.knots_u)
            self.bdv = self.basis_function_derivative(v, self.degree_v, self.knots_v)
            self.bduu = self.basis_function_second_derivative(u, self.degree_u, self.knots_u)
            self.bdvv = self.basis_function_second_derivative(v, self.degree_v, self.knots_v)
            
            # 初始化
            Num = sample_points.shape[0]  # N 下面要用
            weighted_control = wij * control_points
            device = control_points.device
            # 初始化结果张量
            positions = torch.zeros(Num, device=device)
            normals = torch.zeros((Num,3), device=device)
            k1 = torch.zeros(Num, device=device)
            k2 = torch.zeros(Num, device=device)
            J = torch.zeros((Num, 3, 2)) # 雅可比矩阵
            
            for start in range(0, Num, batch_size):
                end = min(start + batch_size, Num)
                current_bu = self.bu[start:end]  # [batch, nu]
                current_bv = self.bv[start:end]  # [batch, nv]
                current_bdu = self.bdu[start:end]  # [batch, nu]
                current_bdv = self.bdv[start:end]  # [batch, nv]
                current_bduu = self.bduu[start:end]  # [batch, nu]
                current_bdvv = self.bdvv[start:end]  # [batch, nv]
                
                # 计算当前批的基函数网格 (不保存完整网格)
                uv_grid = current_bu[:, :, None] * current_bv[:, None, :]  # [batch, nu, nv]
                duv_grid_weight = current_bdu[:, :, None] * current_bv[:, None, :]  # [batch, nu, nv]
                udv_grid_weight = current_bu[:, :, None] * current_bdv[:, None, :]  # [batch, nu, nv]
                duuv_grid_weight = current_bduu[:, :, None] * current_bv[:, None, :]  # [batch, nu, nv]
                dudv_grid_weight = current_bdu[:, :, None] * current_bdv[:, None, :]  # [batch, nu, nv]
                udvv_grid_weight = current_bu[:, :, None] * current_bdvv[:, None, :]  # [batch, nu, nv]
                
                # 计算当前批的分子贡献
                A_batch = torch.einsum('ijk,jk->i', uv_grid, weighted_control)  # [batch]  
                # 计算当前批的分母贡献
                W_batch = torch.einsum('ijk,jk->i', uv_grid, wij)  # [batch]
                Au_batch = torch.einsum('ijk,jk->i', duv_grid_weight, weighted_control)  # [batch]
                Av_batch = torch.einsum('ijk,jk->i', udv_grid_weight, weighted_control)
                Wu_batch = torch.einsum('ijk,jk->i', duv_grid_weight, wij)
                Wv_batch = torch.einsum('ijk,jk->i', udv_grid_weight, wij)
                Auu_batch = torch.einsum('ijk,jk->i', duuv_grid_weight, weighted_control)  # [batch]
                Auv_batch = torch.einsum('ijk,jk->i', dudv_grid_weight, weighted_control)  # [batch]
                Avv_batch = torch.einsum('ijk,jk->i', udvv_grid_weight, weighted_control)
                Wuu_batch = torch.einsum('ijk,jk->i', duuv_grid_weight, wij)
                Wuv_batch = torch.einsum('ijk,jk->i', dudv_grid_weight, wij)
                Wvv_batch = torch.einsum('ijk,jk->i', udvv_grid_weight, wij)

                # 计算法向量
                du = (Au_batch * W_batch - A_batch * Wu_batch) / W_batch**2  # [batch]
                dv = (Av_batch * W_batch - A_batch * Wv_batch) / W_batch**2  # [batch]
                duu = (Auu_batch * W_batch - 2 * Au_batch * Wu_batch - A_batch * Wuu_batch) / W_batch**2 - 2 * Wu_batch**2 * A_batch / W_batch**3  # [batch]
                duv = (Auv_batch * W_batch - Au_batch * Wv_batch - Av_batch * Wu_batch - A_batch * Wuv_batch) / W_batch**2 + 2 * A_batch * Wu_batch * Wv_batch / W_batch**3  # [batch]
                dvv = (Avv_batch * W_batch - 2 * Av_batch * Wv_batch - A_batch * Wvv_batch) / W_batch**2 - 2 * Wv_batch**2 * A_batch / W_batch**3  # [batch]

                # 计算法向量
                du = torch.stack([torch.ones_like(du), torch.zeros_like(du), du], dim=1)  # [batch,3]
                dv = torch.stack([torch.zeros_like(dv), torch.ones_like(dv), dv], dim=1)  # [batch,3]
                duu = torch.stack([torch.zeros_like(duu), torch.zeros_like(duu), duu], dim=1)  # [batch,3]
                duv = torch.stack([torch.zeros_like(duv), torch.zeros_like(duv), duv], dim=1)  # [batch,3]
                dvv = torch.stack([torch.zeros_like(dvv), torch.zeros_like(dvv), dvv], dim=1)  # [batch,3]
                
                # 计算法向量
                normal = torch.cross(du, dv, dim=1)  # [batch,3]
                normal = normal / torch.norm(normal, dim=1, keepdim=True)

                positions[start:end] = A_batch / W_batch  
                normals[start:end] = normal
                J[start:end, :, 0] = du
                J[start:end, :, 1] = dv
                
                # 曲面的第一与第二基本形式
                E = torch.einsum('ij,ij->i', du, du)  # 第一基本形式 E
                F = torch.einsum('ij,ij->i', du, dv)  # 第一基本形式 F
                G = torch.einsum('ij,ij->i', dv, dv)  # 第一基本形式 G
                L = torch.einsum('ij,ij->i', duu, normal)
                M = torch.einsum('ij,ij->i', duv, normal)
                N = torch.einsum('ij,ij->i', dvv, normal)
                K = (L * N - M * M) / (E * G - F * F)  # 高斯曲率
                H = (1.0 / 2) * (L * G + N * E - 2 * M * F) / (E * G - F * F)  # 平均曲率
                k1[start:end] = H + torch.sqrt(H**2 - K)  # 主曲率1
                k2[start:end] = H - torch.sqrt(H**2 - K)
            return positions, normals, k1, k2, J
        
             
    def evaluate_vector2(self, xy_coords, control_points, wij, using_cache:bool = False):
        """
        向量化计算曲面上点的位置和法向量
        Args:
            xy_coords: 采样点坐标
            control_points: 控制点坐标
        """
        with torch.no_grad():
            # 检查是否使用缓存
            if (using_cache == True):
                print("已经改过了。")
                uv_grid_weight = self.cached_uv_grid_weight
                duv_grid_weight = self.cached_duv_grid_weight
                udv_grid_weight = self.cached_udv_grid_weight
            elif (self.cached_sample_points is not None and xy_coords.shape == self.cached_sample_points.shape and torch.allclose(xy_coords, self.cached_sample_points)):
                uv_grid_weight = self.cached_uv_grid_weight
                duv_grid_weight = self.cached_duv_grid_weight
                udv_grid_weight = self.cached_udv_grid_weight
            else:
                u = xy_coords[:, 0]
                v = xy_coords[:, 1]
                basis_u = self.basis_function_vector(u, self.degree_u, self.knots_u)
                basis_v = self.basis_function_vector(v, self.degree_v, self.knots_v)
                basis_u_derivative = self.basis_function_derivative(u, self.degree_u, self.knots_u)
                basis_v_derivative = self.basis_function_derivative(v, self.degree_v, self.knots_v)
                # 直接更新缓存，因为实际上复用的是基函数而不是控制点
                bu = basis_u.unsqueeze(2)
                bv = basis_v.unsqueeze(1)
                bdu = basis_u_derivative.unsqueeze(2)
                bdv = basis_v_derivative.unsqueeze(1)
                self.cached_uv_grid_weight = uv_grid_weight = bu * bv
                self.cached_duv_grid_weight = duv_grid_weight = bdu * bv
                self.cached_udv_grid_weight = udv_grid_weight = bu * bdv

        # 计算位置
        numerator = torch.einsum('ijk,jk->i', uv_grid_weight, wij * control_points)
        denominator = torch.einsum('ijk,jk->i', uv_grid_weight, wij)
        positions = numerator / denominator

        # 计算切向量        
        #du = (torch.einsum('nu,nv,uv->n', basis_u_derivative, basis_v, control_points*wij)*denominator-torch.einsum('nu,nv,uv->n', basis_u_derivative, basis_v, wij)*numerator)/denominator**2  # [N]
        du = (torch.einsum('ijk,jk->i', duv_grid_weight, wij * control_points)*denominator- torch.einsum('ijk,jk->i', duv_grid_weight, wij)*numerator)/denominator**2  # [N]
        #dv = (torch.einsum('nu,nv,uv->n', basis_u, basis_v_derivative, control_points*wij)*denominator-torch.einsum('nu,nv,uv->n', basis_u, basis_v_derivative, wij)*numerator)/denominator**2  # [N]
        dv = (torch.einsum('ijk,jk->i', udv_grid_weight, wij * control_points)*denominator- torch.einsum('ijk,jk->i', udv_grid_weight, wij)*numerator)/denominator**2  # [N]

        # 计算法向量
        du = torch.stack([torch.ones_like(du), torch.zeros_like(du), du], dim=1)  # [N,3]
        dv = torch.stack([torch.zeros_like(dv), torch.ones_like(dv), dv], dim=1)  # [N,3]

        normals = torch.cross(du, dv, dim=1)  # [N,3]
        normals = normals / torch.norm(normals, dim=1, keepdim=True)

        return positions, normals


    def save_to_obj(self, control_points, wij, nu, nv, filename):
        """
        将B样条曲面保存为OBJ文件，生成一个封闭的模型
        参数:
            filename: 输出的obj文件名
            nu: u方向的采样点数量
            nv: v方向的采样点数量
        """
        print("开始写入obj")
        # 生成采样网格
        u = torch.linspace(0, 1, nu) 
        v = torch.linspace(0, 1, nv) 
        u_grid, v_grid = torch.meshgrid(u, v)
        points = torch.stack([u_grid.flatten(), v_grid.flatten()], dim=1).to(self.control_points.device)
        
        # 计算顶面的位置和法向量
        with torch.no_grad():
            top_heights, top_normals = self.evaluate_vector2(points, control_points, wij, using_cache= True)
            #print(top_heights.shape)
        # 将数据转换为numpy数组
        points_np = points.detach().cpu().numpy()
        top_heights_np = top_heights.detach().cpu().numpy()
        top_normals_np = top_normals.detach().cpu().numpy()
        z_min = min(top_heights_np)-0.01
        
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
        
        
if __name__ == "__main__":
    control_points = torch.load("control_points.pt")
    bSurface = NURBS(control_points)
    bSurface.save_to_obj("bSurface.obj",100,100)