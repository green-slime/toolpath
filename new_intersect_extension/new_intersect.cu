#include <torch/extension.h>
#include <iostream>
#include <vector>
#include "geometry_utils.h"
#include "Jet.h"
#include "all_kernels.h"
#include "atomic_utils.h"

using namespace std;

__global__ void compute_intersect_object_parallel_v2(
    float3* d_Ps, ScalarType R, float3* d_path_points, ScalarType MAX_HEIGHT, 
    int points_num, int path_num, int path_len, int Nx, int Ny,  // N是网格大小
    uint64_t* packed_data,  // 改为打包数据
    float3* normals, float3* dirs)
{
    // 每个线程处理一个路径段
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int total_segments = path_num * (path_len - 1);
    if (thread_id >= total_segments) return;
    
    // 计算当前线程对应的路径和段索引
    int path_idx = thread_id / (path_len - 1);
    int seg_idx = thread_id % (path_len - 1);
    
    // 获取线段端点
    int p_idx = path_idx * path_len + seg_idx;
    float3 Pi = d_path_points[p_idx];
    float3 Pj = d_path_points[p_idx + 1];
    
    // 计算路径段的边界框
    float3 box_min = {
        fminf(Pi.x, Pj.x) - R, 
        fminf(Pi.y, Pj.y) - R, 
        fminf(Pi.z, Pj.z) - R
    };
    float3 box_max = {
        fmaxf(Pi.x, Pj.x) + R, 
        fmaxf(Pi.y, Pj.y) + R, 
        fmaxf(Pi.z, Pj.z) + R
    };
    
    // 假设Ps网格在[0,1]×[0,1]，x方向优先
    float grid_step_x = 1.0f / (Nx - 1);
    float grid_step_y = 1.0f / (Ny - 1);
    
    // 将边界框转换为网格索引范围
    int x_min = max(0, (int)floorf(box_min.x / grid_step_x));
    int x_max = min(Nx - 1, (int)ceilf(box_max.x / grid_step_x));
    int y_min = max(0, (int)floorf(box_min.y / grid_step_y));
    int y_max = min(Ny - 1, (int)ceilf(box_max.y / grid_step_y));
    
    // 遍历边界框内的所有网格点
    for (int i = x_min; i <= x_max; i++) {        // x方向
        for (int j = y_min; j <= y_max; j++) {    // y方向
            // 计算线性索引：x方向优先意味着 idx = i * N + j
            int ps_idx = i * Ny + j;
            if (ps_idx >= points_num) continue;
            
            float3 P = d_Ps[ps_idx];
            
            // 精确边界框检查
            if (P.x < box_min.x || P.x > box_max.x || 
                P.y < box_min.y || P.y > box_max.y) continue;
            
            // === 管状表面交点计算 ===
            float3 Di = Pj - Pi;
            float3 Qi = P - Pi;
            
            float Di_norm2 = dot(Di, Di);
            
            // 计算二次方程系数
            float a = Di_norm2 - Di.z * Di.z;
            float b = 2.0f * (-dot(Qi, Di) * Di.z + Qi.z * Di_norm2);
            float c = Di_norm2 * (dot(Qi, Qi) - R * R) - dot(Qi, Di) * dot(Qi, Di);
            
            // 判别式
            float discriminant = b * b - 4.0f * a * c;
            if (discriminant > 0) {
                float sqrt_d = sqrtf(discriminant);
                float s1 = (-b + sqrt_d) / (2.0f * a);
                float s2 = (-b - sqrt_d) / (2.0f * a);
                
                // 检查两个可能的交点
                for (int sol_idx = 0; sol_idx < 2; sol_idx++) {
                    float s = (sol_idx == 0) ? s1 : s2;
                    
                    if (s >= 0.0f) {
                        float t = (dot(Qi, Di) + s * Di.z) / Di_norm2;
                        
                        if (t >= 0.0f && t <= 1.0f) {
                            float new_height = P.z + s;
                            
                            // 原子更新打包数据
                            bool updated = atomicMinPackedData(&packed_data[ps_idx], new_height, 
                                                             (uint16_t)path_idx, (uint16_t)seg_idx);
                        }
                    }
                }
            }
            
            // === 端点球面交点计算 ===
            for (int end_idx = 0; end_idx < 2; end_idx++) {
                float3 endpoint = (end_idx == 0) ? Pi : Pj;
                float3 Q = {P.x - endpoint.x, P.y - endpoint.y, P.z - endpoint.z};
                
                // 检查射线与端点球面的交点
                float d2 = Q.x * Q.x + Q.y * Q.y;
                if (d2 < R * R) {
                    float sqrt_term = sqrtf(R * R - d2);
                    float s = endpoint.z - P.z - sqrt_term;
                    
                    if (s >= 0.0f) {
                        float new_height = P.z + s;
                        
                        bool updated = atomicMinPackedData(&packed_data[ps_idx], new_height, 
                                                         (uint16_t)path_idx, (uint16_t)seg_idx);
                    }  
                }
            }
        }
    }
}

__global__ void unpack_intersect_data(
    uint64_t* packed_data, int points_num,
    float* heights, int* min_indices)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= points_num) return;
    
    PackedIntersectData data;
    data.packed = packed_data[idx];
    
    heights[idx] = data.height;

    min_indices[idx * 3 + 0] = data.path_idx;
    min_indices[idx * 3 + 1] = data.seg_idx;

}

__global__ void compute_intersect_details(
    float3* d_Ps, ScalarType R, float3* d_path_points, 
    int points_num, int path_len, float MAX_HEIGHT,
    int* min_indices, float* heights, float3* normals, float3* dirs)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= points_num) return;
    
    int path_idx = min_indices[idx * 3 + 0];
    int seg_idx = min_indices[idx * 3 + 1];
    
    // 如果没有交点，跳过
    if (path_idx == 0xFFFF || seg_idx == 0xFFFF) return;
    
    float3 P = d_Ps[idx];
    
    // 获取线段端点
    int p_idx = path_idx * path_len + seg_idx;
    float3 Pi = d_path_points[p_idx];
    float3 Pj = d_path_points[p_idx + 1];
    int best_is_endpoint = -1; // 0表示管状表面，1表示Pi端点，2表示Pj端点
    
    float intersection_height = heights[idx];
    float min_s = MAX_HEIGHT - P.z;
    float3 best_normal = {0, 0, 1};
    float3 best_dir = {0, 0, 0};
    
    // 计算路径段信息
    float3 Di = Pj - Pi;
    float3 Qi = P - Pi;
    float Di_norm2 = dot(Di, Di);
    
    // 检查管状表面交点
    float a = Di_norm2 - Di.z * Di.z;
    float b = 2.0f * (-dot(Qi, Di) * Di.z + Qi.z * Di_norm2);
    float c = Di_norm2 * (dot(Qi, Qi) - R * R) - dot(Qi, Di) * dot(Qi, Di);
    
    float discriminant = b * b - 4.0f * a * c;
    if (discriminant > 0) {
        float sqrt_d = sqrtf(discriminant);
        float s1 = (-b + sqrt_d) / (2.0f * a);
        float s2 = (-b - sqrt_d) / (2.0f * a);
        
        for (int sol_idx = 0; sol_idx < 2; sol_idx++) {
            float s = (sol_idx == 0) ? s1 : s2;
            if (s >= 0.0f) {
                float t = (dot(Qi, Di) + s * Di.z) / Di_norm2;
                if (t >= 0.0f && t <= 1.0f) {
                    float3 intersect = {P.x, P.y, P.z + s};
                    float3 center = {Pi.x + t * Di.x, Pi.y + t * Di.y, Pi.z + t * Di.z};
                    float3 normal = {
                        (center.x - intersect.x) / R,
                        (center.y - intersect.y) / R,
                        (center.z - intersect.z) / R
                    };
                    float3 dir = {Pj.x - Pi.x, Pj.y - Pi.y, Pj.z - Pi.z};
                    
                    // 找到更近的交点，更新最小值
                    if (s < min_s) {
                        min_s = s;
                        best_normal = normal;
                        best_dir = dir;
                        best_is_endpoint = 0;  // 这是管状表面的交点
                    }
                }
            }
        }
    }
    
    // 检查端点球面
    for (int end_idx = 0; end_idx < 2; end_idx++) {
        float3 endpoint = (end_idx == 0) ? Pi : Pj;
        float3 Q = {P.x - endpoint.x, P.y - endpoint.y, P.z - endpoint.z};
        
        // 检查射线与端点球面的交点
        float d2 = Q.x * Q.x + Q.y * Q.y;
        if (d2 < R * R) {
            float sqrt_term = sqrtf(R * R - d2);
            float s = endpoint.z - P.z - sqrt_term; // 只考虑向下的交点
            if (s >= 0.0f) {
                // 计算交点和法向量
                float3 intersect = {P.x, P.y, P.z + s};
                float3 normal = {
                    (endpoint.x - intersect.x) / R,
                    (endpoint.y - intersect.y) / R,
                    (endpoint.z - intersect.z) / R
                };
                float3 dir = {Pj.x - Pi.x, Pj.y - Pi.y, Pj.z - Pi.z};
                // 找到更近的端点交点，更新最小值
                if (s < min_s) {
                    min_s = s;
                    best_normal = normal;
                    best_dir = dir;
                    best_is_endpoint = end_idx + 1;  // 这是端点球面的交点，1 为 Pi，2 为 Pj
                }
            }
        }
    }
    if(fabsf(min_s + P.z - intersection_height) > 1e-3) {
        // 说明之前的打包数据有问题
        printf("Data mismatch at idx %d: packed height = %f, computed height = %f\n", \
                idx, intersection_height, P.z + min_s);
    }
    // 写入结果
    best_normal = normalize(best_normal);
    best_dir = normalize(best_dir);
    heights[idx] = P.z + min_s;
    normals[idx] = best_normal;
    dirs[idx] = best_dir;

    // 存储最近交点的索引信息
    min_indices[idx * 3 + 2] = best_is_endpoint;
}

__global__ void init_packed_data(uint64_t* packed_data, int points_num, float MAX_HEIGHT) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= points_num) return;
    
    PackedIntersectData init_data(MAX_HEIGHT, 0xFFFF, 0xFFFF);
    packed_data[idx] = init_data.packed;
}


std::vector<torch::Tensor> intersect_object_parallel_v2(
    torch::Tensor Ps, torch::Tensor path_points, 
    float R, float MAX_HEIGHT)
{
    cudaSetDevice(Ps.get_device());

    int Nx = Ps.size(0);
    int Ny = Ps.size(1);
    int points_num = Nx * Ny;
    int path_num = path_points.size(0);
    int path_len = path_points.size(1);
    int total_segments = path_num * (path_len - 1);
    
    // ===== 简化方案：直接在GPU上初始化 =====
    auto packed_data = torch::zeros({points_num}, 
                                   torch::TensorOptions().dtype(torch::kInt64).device(Ps.device()));
    
    auto normals = torch::full({points_num, 3}, 0.0f, 
                               torch::TensorOptions().dtype(torch::kFloat32).device(Ps.device()));
    normals.select(1, 2).fill_(1.0f);
    auto dirs = torch::full({points_num, 3}, 0.0f, 
                           torch::TensorOptions().dtype(torch::kFloat32).device(Ps.device()));

    // 获取数据指针
    float3* d_Ps = reinterpret_cast<float3*>(Ps.data_ptr<float>());
    float3* d_path_points = reinterpret_cast<float3*>(path_points.data_ptr<float>());
    uint64_t* d_packed_data = reinterpret_cast<uint64_t*>(packed_data.data_ptr<int64_t>());
    float3* d_normals = reinterpret_cast<float3*>(normals.data_ptr<float>());
    float3* d_dirs = reinterpret_cast<float3*>(dirs.data_ptr<float>());
    
    // 初始化打包数据（在GPU上）
    int threadsPerBlock = 256;
    int blocksPerGrid_init = (points_num + threadsPerBlock - 1) / threadsPerBlock;
    
    // 添加初始化核函数
    init_packed_data<<<blocksPerGrid_init, threadsPerBlock>>>(
        d_packed_data, points_num, MAX_HEIGHT
    );
    
    // 启动CUDA核函数 - 现在线程数等于路径段总数
    int blocksPerGrid = (total_segments + threadsPerBlock - 1) / threadsPerBlock;
    
    compute_intersect_object_parallel_v2<<<blocksPerGrid, threadsPerBlock>>>(
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
    
    int blocksPerGrid2 = (points_num + threadsPerBlock - 1) / threadsPerBlock;
    unpack_intersect_data<<<blocksPerGrid2, threadsPerBlock>>>(
        d_packed_data, points_num, d_heights, d_min_indices
    );

    // 计算交点细节
    compute_intersect_details<<<blocksPerGrid2, threadsPerBlock>>>(
        d_Ps, R, d_path_points, points_num, path_len, MAX_HEIGHT,
        d_min_indices, d_heights, d_normals, d_dirs
    );

    cudaDeviceSynchronize();

    return {heights, normals, dirs};
    //return {heights.reshape({N, N}), normals.reshape({N, N, 3}), min_indices.reshape({N, N, 3})};
}