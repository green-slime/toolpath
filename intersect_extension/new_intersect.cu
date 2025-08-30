#include <torch/extension.h>
#include <iostream>
#include <vector>
#include "geometry_utils.h"
#include "Jet.h"

typedef float ScalarType;

using namespace std;

// 保留原子操作的兼容性代码
#if !defined (__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
__device__ double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}
#endif

// CUDA helper for atomicMax on float (if needed)
__device__ float atomicMinFloat(float* addr, float value) {
    int* address_as_i = (int*)addr;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_i, assumed,
                        __float_as_int(fminf(value, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

__global__ void compute_intersect_object_parallel_v2(
    float3* d_Ps, ScalarType R, float3* d_path_points, ScalarType MAX_HEIGHT, 
    int points_num, int path_num, int path_len, int Nx, int Ny,  // N是网格大小
    ScalarType* heights, float3* normals, int* min_indices, float3* dirs)
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
                            // 计算交点和法向量
                            float new_height = P.z + s;
                            
                            // 原子操作比较并更新最小高度
                            float old_height = atomicMinFloat(&heights[ps_idx], new_height);
                            
                            // 如果成功更新了高度，也更新其他信息
                            if (new_height <= old_height) {
                                float3 intersect = {P.x, P.y, new_height};
                                float3 center = {Pi.x + t * Di.x, Pi.y + t * Di.y, Pi.z + t * Di.z};
                                float3 normal = normalize({
                                    (center.x - intersect.x) / R,
                                    (center.y - intersect.y) / R,
                                    (center.z - intersect.z) / R
                                });
                                float3 dir = normalize({Pj.x - Pi.x, Pj.y - Pi.y, Pj.z - Pi.z});
                                
                                normals[ps_idx] = normal;
                                dirs[ps_idx] = dir;
                                min_indices[ps_idx * 3 + 0] = path_idx;
                                min_indices[ps_idx * 3 + 1] = seg_idx;
                                min_indices[ps_idx * 3 + 2] = 0;  // 管状表面
                            }
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
                        
                        // 原子操作更新
                        float old_height = atomicMinFloat(&heights[ps_idx], new_height);
                        
                        if (new_height <= old_height) {
                            float3 intersect = {P.x, P.y, new_height};
                            float3 normal = normalize({
                                (endpoint.x - intersect.x) / R,
                                (endpoint.y - intersect.y) / R,
                                (endpoint.z - intersect.z) / R
                            });
                            float3 dir = normalize({Pj.x - Pi.x, Pj.y - Pi.y, Pj.z - Pi.z});
                            
                            normals[ps_idx] = normal;
                            dirs[ps_idx] = dir;
                            min_indices[ps_idx * 3 + 0] = path_idx;
                            min_indices[ps_idx * 3 + 1] = seg_idx;
                            min_indices[ps_idx * 3 + 2] = end_idx + 1;  // 端点球面
                        }
                    }
                }
            }
        }
    }
}

std::vector<torch::Tensor> intersect_object_parallel_v2(
    torch::Tensor Ps, torch::Tensor path_points, 
    float R, float MAX_HEIGHT, int N) // 新增N参数
{
    // 这里输入[Nx, Ny, 3]的Ps
    // path_points [path_num, path_len, 3]
    cudaSetDevice(Ps.get_device());
    
    int points_num = Ps.size(0) * Ps.size(1);  // Nx * Ny
    int path_num = path_points.size(0);
    int path_len = path_points.size(1);
    int total_segments = path_num * (path_len - 1);
    
    // 分配输出内存
    auto heights = torch::full({points_num}, MAX_HEIGHT, 
                              torch::TensorOptions().dtype(torch::kFloat32).device(Ps.device()));
    auto normals = torch::full({points_num, 3}, 0.0f, 
                               torch::TensorOptions().dtype(torch::kFloat32).device(Ps.device()));
    normals.select(1, 2).fill_(1.0f);  // 设置z分量为1，即(0,0,1)
    auto min_indices = torch::full({points_num, 3}, -1,
    torch::TensorOptions().dtype(torch::kInt32).device(Ps.device()));
    auto dirs = torch::full({points_num, 3}, 0.0f, 
                               torch::TensorOptions().dtype(torch::kFloat32).device(Ps.device()));

    // 获取数据指针
    float3* d_Ps = reinterpret_cast<float3*>(Ps.data_ptr<float>());
    float3* d_path_points = reinterpret_cast<float3*>(path_points.data_ptr<float>());
    float* d_heights = heights.data_ptr<float>();
    float3* d_normals = reinterpret_cast<float3*>(normals.data_ptr<float>());
    int* d_min_indices = min_indices.data_ptr<int>();
    float3* d_dirs = reinterpret_cast<float3*>(dirs.data_ptr<float>());
    
    // 启动CUDA核函数 - 现在线程数等于路径段总数
    int threadsPerBlock = 256;
    int blocksPerGrid = (total_segments + threadsPerBlock - 1) / threadsPerBlock;
    
    compute_intersect_object_parallel_v2<<<blocksPerGrid, threadsPerBlock>>>(
        d_Ps, R, d_path_points, MAX_HEIGHT, points_num, path_num, path_len, Nx, Ny,
        d_heights, d_normals, d_min_indices, d_dirs
    );

    cudaDeviceSynchronize();

    return {heights, normals, dirs};
    //return {heights.reshape({N, N}), normals.reshape({N, N, 3}), min_indices.reshape({N, N, 3})};
}