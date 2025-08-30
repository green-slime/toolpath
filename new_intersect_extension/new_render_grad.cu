#include <torch/extension.h>
#include <iostream>
#include <vector>
#include "geometry_utils.h"
#include "Jet.h"
#include "all_kernels.h"
#include "atomic_utils.h"

using namespace std;

__global__ void compute_gradients_for_render(
    float3* d_Ps, float R, float3* d_path_points, float MAX_HEIGHT,
    int points_num, int path_num, int path_len, 
    int* min_indices, float z_plane_height, float n1, float n2,
    ScalarType* d_grad_render_to_yj, ScalarType* d_grad_render_to_path_points
) 
{
    // 这个核函数输出这样的一件事：
    // 我们需要知道，loss 对于路径点 i 的梯度
    // 那么由链式法则，因为 loss 对于投影点 yj 的梯度已知（轮子），我们只需要计算点 yj 对于路径点 i 的梯度即可
    // 但是我们不能直接输出对应的 Jacobi 矩阵，因为规模太大了。
    // 因此，我们以类似稀疏矩阵的方式进行记录。我们已经可知一个 yj 只受到一个 i（端点）或两个 i（管道）影响。
    // 我们维护四个数组进行记录，col1, col2, val1, val2
    // col1, col2 记录对于每个 j，i 的索引，val1, val2 记录对应的梯度
    // 若为球，则 col2 为 -1，val2 为 NAN（随意）
    // val1[k] 大小为 4，为[xj.grad[1], xj.grad[2], yj.grad[1], yj.grad[2]]，其中 xj 为投影点的 x 坐标，yj 为投影点的 y 坐标
    // val2[k] 大小为 4，为[xj.grad[4], xj.grad[5], yj.grad[4], yj.grad[5]]，其中 xj 为投影点的 x 坐标，yj 为投影点的 y 坐标

    // 考虑直接改为计算 grad_render_to_path_points
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // 相当于 j
    if (idx >= points_num) return;

    float3 P = d_Ps[idx];
    int best_path_idx = min_indices[idx * 3];
    int best_segment_idx = min_indices[idx * 3 + 1];
    int best_is_endpoint = min_indices[idx * 3 + 2];

    // 检查有效性
    if (best_path_idx == -1 || best_segment_idx == -1 || best_is_endpoint == -1) {
        return;
    }

    Jet s_jet(0.0f);
    Jet3 normal_jet(Jet(0.0f), Jet(0.0f), Jet(1.0f));
    Jet3 height_jet(Jet(P.x), Jet(P.y), Jet(MAX_HEIGHT));
    Jet3 refracted_dir(Jet(0.0f), Jet(0.0f), Jet(1.0f));

    if (best_is_endpoint == 0) { // 管状体交点
        int base_idx = best_path_idx * path_len + best_segment_idx; // 相当于 i
        float3 Pi_host = d_path_points[base_idx];
        float3 Pj_host = d_path_points[base_idx + 1];

        // 创建独立变量（Pi 和 Pj 的 xyz）
        Jet Pi[3], Pj[3];
        // x 不为变量
        Pi[0] = Jet(Pi_host.x, 0); // 梯度索引 0
        Pi[1] = Jet(Pi_host.y, 1); // 梯度索引 1
        Pi[2] = Jet(Pi_host.z, 2); // 梯度索引 2
        Pj[0] = Jet(Pj_host.x, 3); // 梯度索引 3
        Pj[1] = Jet(Pj_host.y, 4); // 梯度索引 4
        Pj[2] = Jet(Pj_host.z, 5); // 梯度索引 5

        Jet3 Pi_jet(Pi[0], Pi[1], Pi[2]);
        Jet3 Pj_jet(Pj[0], Pj[1], Pj[2]);

        // 计算几何量
        Jet3 Di = Pj_jet - Pi_jet;
        Jet3 Qi = Jet3(P) - Pi_jet;
        Jet Di_norm2 = dot(Di, Di);
        
        // 求解二次方程
        Jet a = Di_norm2 - Di.z * Di.z;
        Jet b = Jet(2.0f) * (-dot(Qi, Di) * Di.z + Qi.z * Di_norm2);
        Jet c = Di_norm2 * (dot(Qi, Qi) - R * R) - pow2(dot(Qi, Di));
        
        Jet discriminant = b * b - Jet(4.0f) * a * c;
        Jet sqrt_d = sqrt(discriminant);
        Jet s1 = (-b + sqrt_d) / (Jet(2.0f) * a);
        Jet s2 = (-b - sqrt_d) / (Jet(2.0f) * a);
        
        // 检查有效解
        Jet t1 = (dot(Qi, Di) + s1 * Di.z) / Di_norm2;
        Jet t2 = (dot(Qi, Di) + s2 * Di.z) / Di_norm2;
        
        if (s2.val >= 0 && t2.val >= 0 && t2.val <= 1) {
            s_jet = s2;
            Jet3 center = Pi_jet + t2 * Di;
            Jet3 intersect = {Jet(P.x), Jet(P.y), Jet(P.z) + s2};
            normal_jet = (center - intersect) / Jet(R);
            height_jet = intersect;
        } 
        else if (s1.val >= 0 && t1.val >= 0 && t1.val <= 1) {
            s_jet = s1;
            Jet3 center = Pi_jet + t1 * Di;
            Jet3 intersect = {Jet(P.x), Jet(P.y), Jet(P.z) + s1};
            normal_jet = (center - intersect) / Jet(R);
            height_jet = intersect;
        } 
        
        // 归一化法向量
        normal_jet = normalize(normal_jet);
        
    } else if(best_is_endpoint > 0){ // 端点球面交点
        int p_idx = best_path_idx * path_len + best_segment_idx;
        if (best_is_endpoint == 2) p_idx += 1; // 终点
        
        float3 P_end_host = d_path_points[p_idx];
        Jet P_end[3];
        P_end[0] = Jet(P_end_host.x, 0); // 梯度索引 0
        P_end[1] = Jet(P_end_host.y, 1); // 梯度索引 1
        P_end[2] = Jet(P_end_host.z, 2); // 梯度索引 2
        
        Jet3 endpoint(P_end[0], P_end[1], P_end[2]);
        
        // 计算球面交点
        Jet3 Q = Jet3(P) - endpoint;
        Jet d2 = Q.x * Q.x + Q.y * Q.y;
        Jet sqrt_term = sqrt(Jet(R*R) - d2);
        s_jet = endpoint.z - Jet(P.z) - sqrt_term;
        
        // 计算法向量
        Jet3 intersect = {Jet(P.x), Jet(P.y), Jet(P.z) + s_jet};
        normal_jet = (endpoint - intersect) / R;
        normal_jet = normalize(normal_jet);
        height_jet = intersect;
    }

    // 计算折射
    Jet3 incident_dir = {Jet(0.0f), Jet(0.0f), Jet(1.0f)};
    Jet cos_i = dot(incident_dir, normal_jet);
    Jet n = Jet(n1/n2);
    Jet cos_t_pw2 = Jet(1.0f) - n * n * (Jet(1.0f) - cos_i * cos_i);
    if (cos_t_pw2.val < 0.0f){
        Jet cos_t = Jet(0.0f); // 梯度也同时置 0
        refracted_dir = incident_dir + Jet(2.0f) * cos_i * normal_jet;
    }
    else{
        Jet cos_t = sqrt(cos_t_pw2);
        refracted_dir = n * incident_dir - (n * cos_i - cos_t) * normal_jet;
        refracted_dir = normalize(refracted_dir);
    }
    Jet t = (Jet(z_plane_height) - height_jet.z) / refracted_dir.z;
    Jet3 receiver_point = height_jet + t * refracted_dir;
    Jet xj = receiver_point.x;
    Jet yj = receiver_point.y;

    // 写入梯度 - 需要修改为包含x分量
    if(best_is_endpoint == 0){
        // 管道 - 现在需要写入x, y, z三个分量的梯度
        int base_idx = best_path_idx * path_len + best_segment_idx;
        // 累加 loss 的梯度 - Pi点的梯度
        atomicAdd(&d_grad_render_to_path_points[base_idx * 3 + 0],  // x分量
                  xj.grad[0] * d_grad_render_to_yj[idx * 2 + 0] + yj.grad[0] * d_grad_render_to_yj[idx * 2 + 1]);
        atomicAdd(&d_grad_render_to_path_points[base_idx * 3 + 1],  // y分量
                  xj.grad[1] * d_grad_render_to_yj[idx * 2 + 0] + yj.grad[1] * d_grad_render_to_yj[idx * 2 + 1]);
        atomicAdd(&d_grad_render_to_path_points[base_idx * 3 + 2],  // z分量
                  xj.grad[2] * d_grad_render_to_yj[idx * 2 + 0] + yj.grad[2] * d_grad_render_to_yj[idx * 2 + 1]);
        // Pj点的梯度
        atomicAdd(&d_grad_render_to_path_points[(base_idx + 1) * 3 + 0],  // x分量
                  xj.grad[3] * d_grad_render_to_yj[idx * 2 + 0] + yj.grad[3] * d_grad_render_to_yj[idx * 2 + 1]);
        atomicAdd(&d_grad_render_to_path_points[(base_idx + 1) * 3 + 1],  // y分量
                  xj.grad[4] * d_grad_render_to_yj[idx * 2 + 0] + yj.grad[4] * d_grad_render_to_yj[idx * 2 + 1]);
        atomicAdd(&d_grad_render_to_path_points[(base_idx + 1) * 3 + 2],  // z分量
                  xj.grad[5] * d_grad_render_to_yj[idx * 2 + 0] + yj.grad[5] * d_grad_render_to_yj[idx * 2 + 1]);
    }   
    else if(best_is_endpoint > 0){
        // 端点 - 现在需要写入x, y, z三个分量的梯度
        int p_idx = best_path_idx * path_len + best_segment_idx;
        if (best_is_endpoint == 2) p_idx += 1; // 终点
        // 累加 loss 的梯度
        atomicAdd(&d_grad_render_to_path_points[p_idx * 3 + 0],  // x分量
                  xj.grad[0] * d_grad_render_to_yj[idx * 2 + 0] + yj.grad[0] * d_grad_render_to_yj[idx * 2 + 1]);
        atomicAdd(&d_grad_render_to_path_points[p_idx * 3 + 1],  // y分量
                  xj.grad[1] * d_grad_render_to_yj[idx * 2 + 0] + yj.grad[1] * d_grad_render_to_yj[idx * 2 + 1]);
        atomicAdd(&d_grad_render_to_path_points[p_idx * 3 + 2],  // z分量
                  xj.grad[2] * d_grad_render_to_yj[idx * 2 + 0] + yj.grad[2] * d_grad_render_to_yj[idx * 2 + 1]);
    }
}

// Python接口函数
std::vector<torch::Tensor> intersect_with_grad_for_render(
    torch::Tensor Ps, torch::Tensor path_points, torch::Tensor grad_render_to_yj, 
    float R, float MAX_HEIGHT, float n1, float n2, float z_plane_height)
{
    // Ps: 射线起点，形状 (Nx, Ny, 3)，[Px, Py, Pz]，固定常量
    // path_points: 路径点，形状 (path_num, path_len, 3)，每个路径点的 [x, y, z] 坐标
    // 需要预先使用 contiguous() 确保数据在连续内存中
    // grad_render_to_yj: 形状 (Nx*Ny, 2)，每个射线的梯度 [dL/dxj, dL/dyj]，其中 xj, yj 是投影点的坐标
    // 返回：grad_render_to_path_points：形状 (path_num, path_len, 3)，每个路径点的梯度 [dL/dxi, dL/dyi, dL/dzi]，其中 yi, zi 是变量 i 的 y, z 坐标
    cudaSetDevice(Ps.get_device());

    int Nx = Ps.size(0);
    int Ny = Ps.size(1);
    int points_num = Nx * Ny;
    int path_num = path_points.size(0);
    int path_len = path_points.size(1);
    int total_segments = path_num * (path_len - 1);

    float3* d_Ps = reinterpret_cast<float3*>(Ps.data_ptr<float>());
    float3* d_path_points = reinterpret_cast<float3*>(path_points.data_ptr<float>());
    ScalarType* d_grad_render_to_yj = grad_render_to_yj.data_ptr<ScalarType>(); // [Nx*Ny,2]

    // ===== 使用打包数据结构 =====
    auto packed_data = torch::zeros({points_num}, 
                                   torch::TensorOptions().dtype(torch::kInt64).device(Ps.device()));
    
    auto normals = torch::full({points_num, 3}, 0.0f, 
                               torch::TensorOptions().dtype(torch::kFloat32).device(Ps.device()));
    normals.select(1, 2).fill_(1.0f);
    auto dirs = torch::full({points_num, 3}, 0.0f, 
                           torch::TensorOptions().dtype(torch::kFloat32).device(Ps.device()));

    auto grad_render_to_path_points = torch::zeros({path_num, path_len, 3}, 
                                                  torch::TensorOptions().dtype(torch::kFloat32).device(Ps.device()));

    // 获取数据指针
    uint64_t* d_packed_data = reinterpret_cast<uint64_t*>(packed_data.data_ptr<int64_t>());
    float3* d_normals = reinterpret_cast<float3*>(normals.data_ptr<float>());
    float3* d_dirs = reinterpret_cast<float3*>(dirs.data_ptr<float>());
    ScalarType* d_grad_render_to_path_points = grad_render_to_path_points.data_ptr<ScalarType>();

    int threadsPerBlock = 256;
    
    // 初始化打包数据
    int blocksPerGrid_init = (points_num + threadsPerBlock - 1) / threadsPerBlock;
    init_packed_data<<<blocksPerGrid_init, threadsPerBlock>>>(
        d_packed_data, points_num, MAX_HEIGHT
    );

    // 计算交点（对路径段并行）
    int blocksPerGrid_segments = (total_segments + threadsPerBlock - 1) / threadsPerBlock;
    compute_intersect_object_parallel_v2<<<blocksPerGrid_segments, threadsPerBlock>>>(
        d_Ps, R, d_path_points, MAX_HEIGHT, points_num, path_num, path_len, Nx, Ny,
        d_packed_data, d_normals, d_dirs
    );

    // 解包数据
    auto heights = torch::zeros({points_num}, 
                               torch::TensorOptions().dtype(torch::kFloat32).device(Ps.device()));
    auto min_indices = torch::full({points_num, 3}, -1,
                                  torch::TensorOptions().dtype(torch::kInt32).device(Ps.device()));
    
    float* d_heights = heights.data_ptr<float>();
    int* d_min_indices = min_indices.data_ptr<int>();
    
    int blocksPerGrid_points = (points_num + threadsPerBlock - 1) / threadsPerBlock;
    unpack_intersect_data<<<blocksPerGrid_points, threadsPerBlock>>>(
        d_packed_data, points_num, d_heights, d_min_indices
    );
    compute_intersect_details<<<blocksPerGrid_points, threadsPerBlock>>>(
        d_Ps, R, d_path_points, points_num, path_len, MAX_HEIGHT,
        d_min_indices, d_heights, d_normals, d_dirs
    );

    // 计算渲染梯度（对采样点并行）
    compute_gradients_for_render<<<blocksPerGrid_points, threadsPerBlock>>>(
        d_Ps, R, d_path_points, MAX_HEIGHT, points_num, path_num, path_len, 
        d_min_indices, z_plane_height, n1, n2,
        d_grad_render_to_yj, d_grad_render_to_path_points
    );

    cudaDeviceSynchronize();

    return {grad_render_to_path_points};
}