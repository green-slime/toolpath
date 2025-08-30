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
__device__ float atomicMaxFloat(float* addr, float value) {
    int* address_as_i = (int*)addr;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_i, assumed,
                        __float_as_int(fmaxf(value, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

__global__ void compute_intersect(float3* d_Ps, ScalarType R, float3* d_path_points, ScalarType MAX_HEIGHT, int points_num, int path_num, int path_len, ScalarType* heights, float3* normals, int* min_indices, float3* dirs, bool check_flag)
{
    // intersect.py 中主要函数的CUDA实现
    // Ps: 射线起点，形状 (N, 3)，[Px, Py, Pz]，固定常量
    // path_points: 路径点，形状 (path_num, path_len, 3)，每个路径点的 [x, y, z] 坐标
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= points_num) return;
    float3 P = d_Ps[idx];
    
    // 合并管状和球体，以一个小包络面为单位
    // 用于跟踪最近交点
    ScalarType min_s = MAX_HEIGHT - P.z;
    float3 best_normal = {0.0f, 0.0f, 1.0f};
    float3 best_dir = {1.0f, 0.0f, 0.0f}; 
    int best_path_idx = -1;
    int best_segment_idx = -1;
    int best_is_endpoint = -1;  // 0=不是端点, 1=是端点

    // 遍历所有路径
    for (int path_idx = 0; path_idx < path_num; path_idx++) {
        // 处理每个线段（包括端点）
        for (int seg_idx = 0; seg_idx < path_len - 1; seg_idx++) {
            // 获取线段端点
            int p_idx = path_idx * path_len + seg_idx;
            float3 Pi = d_path_points[p_idx];
            float3 Pj = d_path_points[p_idx + 1];

            // 计算包含管状和端点球面的整体边界框
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

            // 射线与边界框的 XY 平面交点检查
            if (P.x >= box_min.x && P.x <= box_max.x && 
                P.y >= box_min.y && P.y <= box_max.y) {
                    // 计算方向向量和相对向量
                float3 Di = Pj - Pi;
                float3 Qi = P - Pi;
                
                float Di_norm2 = dot(Di, Di);
                
                // 计算二次方程系数
                float a = Di_norm2 - Di.z * Di.z; // Di.x 即为 x 方向采样间距，必不为 0. 故 Di_norm2 和 a 都不会为 0。
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
                            /* if(path_idx==246&&seg_idx==400&&check_flag==true&&idx==427261){
                                printf("t=%.9f, s=%.9f\n", t, s);
                                printf("a=%.16f, b=%.16f, c=%.16f, d=%.16f, sqrt_d=%.16f\n",a,b,c,discriminant,sqrt_d);
                                //printf("Qi:(%.9f, %.9f, %.9f), Di:(%.9f, %.9f, %.9f), Di_norm2: %.9f\n", Qi.x, Qi.y, Qi.z, Di.x, Di.y, Di.z, Di_norm2);
                            } */
                            if (t >= 0.0f && t <= 1.0f) {
                                // 计算交点和法向量
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
                                    best_path_idx = path_idx;
                                    best_segment_idx = seg_idx;
                                    best_is_endpoint = 0;  // 这是管状表面的交点
                                }
                            }
                        }
                    }
                }
                // 检查端点球面交点
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
                                best_path_idx = path_idx;
                                best_segment_idx = seg_idx;
                                best_is_endpoint = end_idx + 1;  // 这是端点球面的交点，1 为 Pi，2 为 Pj
                            }
                        }
                    }
                }
            }
        }
    }
    // 写入结果
    best_normal = normalize(best_normal);
    best_dir = normalize(best_dir);
    heights[idx] = P.z + min_s;
    normals[idx] = best_normal;
    dirs[idx] = best_dir;
    
    // 存储最近交点的索引信息
    min_indices[idx * 3 + 0] = best_path_idx;
    min_indices[idx * 3 + 1] = best_segment_idx;
    min_indices[idx * 3 + 2] = best_is_endpoint;

}

// 利用 Jet 类进行自动微分计算梯度

__global__ void compute_gradients(
    float3* d_Ps, float R, float3* d_path_points, float MAX_HEIGHT,
    int points_num, int path_num, int path_len, float3* d_target_pos,
    int* min_indices, float z_plane_height, float n1, float n2,
    ScalarType* d_loss, // 输出每个光束产生的 loss
    ScalarType* d_grad, // 输出 loss 之和对各变量的梯度
    ScalarType* grad_s_path_points,   // 输出：s 关于路径点的梯度
    ScalarType* grad_n_path_points    // 输出：法向量关于路径点的梯度 [path_num * path_len * 3 * 3]
    // 法向量: ijkl 其中 i 是路径索引，j 是路径点索引，k 是法向量分量 (nx, ny, nz)，l 是路径点分量 (x, y, z
) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= points_num) return;

    float3 P = d_Ps[idx];
    int best_path_idx = min_indices[idx * 3];
    int best_segment_idx = min_indices[idx * 3 + 1];
    int best_is_endpoint = min_indices[idx * 3 + 2];
    float3 target_pos = d_target_pos[idx];

    Jet s_jet(0.0f);
    Jet3 normal_jet(Jet(0.0f), Jet(0.0f), Jet(1.0f));
    Jet3 height_jet(Jet(P.x), Jet(P.y), Jet(MAX_HEIGHT));
    Jet3 refracted_dir(Jet(0.0f), Jet(0.0f), Jet(1.0f));
    Jet3 target_pos_jet(Jet(target_pos.x), Jet(target_pos.y), Jet(target_pos.z));

    if (best_is_endpoint == 0) { // 管状体交点
        int base_idx = best_path_idx * path_len + best_segment_idx;
        float3 Pi_host = d_path_points[base_idx];
        float3 Pj_host = d_path_points[base_idx + 1];

        // 创建独立变量（Pi 和 Pj 的 xyz）
        Jet Pi[3], Pj[3];
        // x 不为变量
        Pi[0] = Jet(Pi_host.x); // 梯度索引 0
        Pi[1] = Jet(Pi_host.y, 1); // 梯度索引 1
        Pi[2] = Jet(Pi_host.z, 2); // 梯度索引 2
        Pj[0] = Jet(Pj_host.x); // 梯度索引 3
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
        
        /* // 累加 s 的梯度
        for (int i = 0; i < 3; ++i) {
            atomicAdd(&grad_s_path_points[base_idx * 3 + i], s_jet.grad[i]);
            atomicAdd(&grad_s_path_points[(base_idx+1) * 3 + i], s_jet.grad[i+3]);
        } */
        
    } else if(best_is_endpoint != -1){ // 端点球面交点
        int p_idx = best_path_idx * path_len + best_segment_idx;
        if (best_is_endpoint == 2) p_idx += 1; // 终点
        
        float3 P_end_host = d_path_points[p_idx];
        Jet P_end[3];
        P_end[0] = Jet(P_end_host.x); // 梯度索引 0
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
        
        /* // 累加 s 的梯度
        for (int i = 0; i < 3; ++i) {
            atomicAdd(&grad_s_path_points[p_idx * 3 + i], s_jet.grad[i]);
        } */
    }
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
    Jet loss = norm2(receiver_point - target_pos_jet);

    atomicAdd(&d_loss[idx], loss.val);

    // 写入梯度
    if(best_is_endpoint == 0){
        // 管道
        int base_idx = best_path_idx * path_len + best_segment_idx;
        // 累加 loss 的梯度
        for (int i = 0; i < 3; ++i) {
            atomicAdd(&d_grad[base_idx * 3 + i], loss.grad[i]);
            atomicAdd(&d_grad[(base_idx+1) * 3 + i], loss.grad[i+3]);
        }
    }
    else if(best_is_endpoint > 0){
        // 端点
        int p_idx = best_path_idx * path_len + best_segment_idx;
        if (best_is_endpoint == 2) p_idx += 1; // 终点
        // 累加 loss 的梯度
        for (int i = 0; i < 3; ++i) {
            atomicAdd(&d_grad[p_idx * 3 + i], loss.grad[i]);
        }
    }
    
    
}

__global__ void compute_gradients_for_normals(
    float3* d_Ps, float R, float3* d_path_points, float MAX_HEIGHT,
    int points_num, int path_num, int path_len, float3* d_target_normals,
    int* min_indices, float z_plane_height, float n1, float n2,
    ScalarType* d_loss, // 输出每个光束产生的 loss
    ScalarType* d_grad // 输出 loss 之和对各变量的梯度
) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= points_num) return;

    float3 P = d_Ps[idx];
    int best_path_idx = min_indices[idx * 3];
    int best_segment_idx = min_indices[idx * 3 + 1];
    int best_is_endpoint = min_indices[idx * 3 + 2];
    float3 target_normal = d_target_normals[idx];

    Jet s_jet(0.0f);
    Jet3 normal_jet(Jet(0.0f), Jet(0.0f), Jet(1.0f));
    Jet3 height_jet(Jet(P.x), Jet(P.y), Jet(MAX_HEIGHT));
    Jet3 refracted_dir(Jet(0.0f), Jet(0.0f), Jet(1.0f));
    Jet3 target_normal_jet(Jet(target_normal.x), Jet(target_normal.y), Jet(target_normal.z));

    if (best_is_endpoint == 0) { // 管状体交点
        int base_idx = best_path_idx * path_len + best_segment_idx;
        float3 Pi_host = d_path_points[base_idx];
        float3 Pj_host = d_path_points[base_idx + 1];

        // 创建独立变量（Pi 和 Pj 的 xyz）
        Jet Pi[3], Pj[3];
        // x 不为变量
        Pi[0] = Jet(Pi_host.x); // 梯度索引 0
        Pi[1] = Jet(Pi_host.y, 1); // 梯度索引 1
        Pi[2] = Jet(Pi_host.z, 2); // 梯度索引 2
        Pj[0] = Jet(Pj_host.x); // 梯度索引 3
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
        
        /* // 累加 s 的梯度
        for (int i = 0; i < 3; ++i) {
            atomicAdd(&grad_s_path_points[base_idx * 3 + i], s_jet.grad[i]);
            atomicAdd(&grad_s_path_points[(base_idx+1) * 3 + i], s_jet.grad[i+3]);
        } */
        
    } else if(best_is_endpoint != -1){ // 端点球面交点
        int p_idx = best_path_idx * path_len + best_segment_idx;
        if (best_is_endpoint == 2) p_idx += 1; // 终点
        
        float3 P_end_host = d_path_points[p_idx];
        Jet P_end[3];
        P_end[0] = Jet(P_end_host.x); // 梯度索引 0
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
        
        /* // 累加 s 的梯度
        for (int i = 0; i < 3; ++i) {
            atomicAdd(&grad_s_path_points[p_idx * 3 + i], s_jet.grad[i]);
        } */
    }
    Jet loss = norm2(normal_jet - target_normal_jet);

    atomicAdd(&d_loss[idx], loss.val);

    // 写入梯度
    if(best_is_endpoint == 0){
        // 管道
        int base_idx = best_path_idx * path_len + best_segment_idx;
        // 累加 loss 的梯度
        for (int i = 0; i < 3; ++i) {
            atomicAdd(&d_grad[base_idx * 3 + i], loss.grad[i]);
            atomicAdd(&d_grad[(base_idx+1) * 3 + i], loss.grad[i+3]);
        }
    }
    else if(best_is_endpoint > 0){
        // 端点
        int p_idx = best_path_idx * path_len + best_segment_idx;
        if (best_is_endpoint == 2) p_idx += 1; // 终点
        // 累加 loss 的梯度
        for (int i = 0; i < 3; ++i) {
            atomicAdd(&d_grad[p_idx * 3 + i], loss.grad[i]);
        }
    }
    
    
}

__global__ void compute_gradients_for_heights(
    float3* d_Ps, float R, float3* d_path_points, float MAX_HEIGHT,
    int points_num, int path_num, int path_len, 
    int* min_indices, 
    ScalarType* d_target_height, // 目标高度
    ScalarType* d_loss, // 输出每个光束产生的 loss
    ScalarType* d_grad, // 输出 loss 之和对各变量的梯度
    ScalarType* d_heights_for_check, ScalarType* d_heights
) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= points_num) return;

    float3 P = d_Ps[idx];
    //ScalarType height = d_heights[idx];
    ScalarType target_height = d_target_height[idx];
    int best_path_idx = min_indices[idx * 3];
    int best_segment_idx = min_indices[idx * 3 + 1];
    int best_is_endpoint = min_indices[idx * 3 + 2];

    Jet s_jet(MAX_HEIGHT - P.z);
    Jet height_jet(MAX_HEIGHT);

    //Jet s1(0.0f); Jet s2(0.0f); Jet t1(0.0f); Jet t2(0.0f);
    //float fa,fb,fc,fd,fsqrtd;
    //float qx,qy,qz,dx,dy,dz,d2;

    if (best_is_endpoint == 0) { // 管状体交点
        int base_idx = best_path_idx * path_len + best_segment_idx;
        float3 Pi_host = d_path_points[base_idx];
        float3 Pj_host = d_path_points[base_idx + 1];

        // 创建独立变量（Pi 和 Pj 的 xyz）
        Jet Pi[3], Pj[3];
        // x 不为变量
        Pi[0] = Jet(Pi_host.x); // 梯度索引 0
        Pi[1] = Jet(Pi_host.y, 1); // 梯度索引 1
        Pi[2] = Jet(Pi_host.z, 2); // 梯度索引 2
        Pj[0] = Jet(Pj_host.x); // 梯度索引 3
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

        //fa=a.val;fb=b.val;fc=c.val;fd=discriminant.val;fsqrtd=sqrt_d.val;
        //qx=Qi.x.val;qy=Qi.y.val;qz=Qi.z.val;dx=Di.x.val;dy=Di.y.val;dz=Di.z.val;d2=Di_norm2.val;
        
        // 检查有效解
        Jet t1 = (dot(Qi, Di) + s1 * Di.z) / Di_norm2;
        Jet t2 = (dot(Qi, Di) + s2 * Di.z) / Di_norm2;
        
        if (s2.val >= 0.0f && t2.val >= 0.0f && t2.val <= 1.0f) {
            s_jet = s2;
        } 
        else if (s1.val >= 0.0f && t1.val >= 0.0f && t1.val <= 1.0f) {
            s_jet = s1;
        } 
        height_jet = Jet(P.z) + s_jet; // 计算高度
        
    } else if(best_is_endpoint > 0){ // 端点球面交点
        int p_idx = best_path_idx * path_len + best_segment_idx;
        if (best_is_endpoint == 2) p_idx += 1; // 终点
        
        float3 P_end_host = d_path_points[p_idx];
        Jet P_end[3];
        P_end[0] = Jet(P_end_host.x); // 梯度索引 0
        P_end[1] = Jet(P_end_host.y, 1); // 梯度索引 1
        P_end[2] = Jet(P_end_host.z, 2); // 梯度索引 2
        
        Jet3 endpoint(P_end[0], P_end[1], P_end[2]);
        
        // 计算球面交点
        Jet3 Q = Jet3(P) - endpoint;
        Jet d2 = Q.x * Q.x + Q.y * Q.y;
        Jet sqrt_term = sqrt(Jet(R*R) - d2);
        s_jet = endpoint.z - Jet(P.z) - sqrt_term;
        height_jet = Jet(P.z) + s_jet; // 计算高度
    }
    Jet loss(0.0f);
    /* d_heights_for_check[idx] = height_jet.val;
    if(fabsf(height-height_jet.val)>0.1){
        printf("idx: %d\n", idx);
        printf("height: %.9f, computed: %.9f, best_path_idx: %d, best_segment_idx: %d, best_is_endpoint: %d\n", height, height_jet.val, best_path_idx, best_segment_idx, best_is_endpoint);
        if(best_is_endpoint==0){
            printf("s1: %.9f, s2: %.9f, t1: %.9f, t2: %.9f\n", s1.val, s2.val, t1.val, t2.val);
            printf("a=%.16f, b=%.16f, c=%.16f, d=%.16f, sqrt_d=%.16f\n",fa,fb,fc,fd,fsqrtd);
            //printf("Qi:(%.9f, %.9f, %.9f), Di:(%.9f, %.9f, %.9f), Di_norm2: %.9f\n", qx, qy, qz, dx, dy, dz, d2);
        }
    } */
    if(height_jet.val > target_height){
        // 如果高度大于目标高度，计算损失
        loss = (height_jet - Jet(target_height)) * (height_jet - Jet(target_height)); // 使用平方损失
    }
    else{
        // 如果高度小于等于目标高度，加大惩罚
        loss = Jet(5.0f) * (Jet(target_height) - height_jet) * (Jet(target_height) - height_jet); // 使用平方损失
    }

    atomicAdd(&d_loss[idx], loss.val);

    // 写入梯度
    if(best_is_endpoint == 0){
        // 管道
        int base_idx = best_path_idx * path_len + best_segment_idx;
        // 累加 loss 的梯度
        for (int i = 0; i < 3; ++i) {
            atomicAdd(&d_grad[base_idx * 3 + i], loss.grad[i]);
            atomicAdd(&d_grad[(base_idx+1) * 3 + i], loss.grad[i+3]);
        }
    }
    else if(best_is_endpoint > 0){
        // 端点
        int p_idx = best_path_idx * path_len + best_segment_idx;
        if (best_is_endpoint == 2) p_idx += 1; // 终点
        // 累加 loss 的梯度
        for (int i = 0; i < 3; ++i) {
            atomicAdd(&d_grad[p_idx * 3 + i], loss.grad[i]);
        }
    }
    
    
}

//================================================================================================================================================================


// Python接口函数
std::vector<torch::Tensor> intersect(torch::Tensor Ps, torch::Tensor path_points, float R, float MAX_HEIGHT, bool output_dirs)
{
    // Ps: 射线起点，形状 (N, 3)，[Px, Py, Pz]，固定常量
    // path_points: 路径点，形状 (path_num, path_len, 3)，每个路径点的 [x, y, z] 坐标
    // 需要预先使用 contiguous() 确保数据在连续内存中
    cudaSetDevice(Ps.get_device());
    int threadnum = 256;

    int points_num = Ps.size(0);
    int path_num = path_points.size(0);
    int path_len = path_points.size(1);

    float3* d_Ps = (float3*)Ps.data_ptr<ScalarType>();
    float3* d_path_points = (float3*)path_points.data_ptr<ScalarType>();

    ScalarType* d_heights;
    cudaMalloc(&d_heights, points_num * sizeof(ScalarType));
    cudaMemset(d_heights, 0, points_num * sizeof(ScalarType));

    float3* d_normals;
    cudaMalloc(&d_normals, points_num * sizeof(float3));
    cudaMemset(d_normals, 0, points_num * sizeof(float3));

    float3* d_dirs;
    cudaMalloc(&d_dirs, points_num * sizeof(float3));
    cudaMemset(d_dirs, 0, points_num * sizeof(float3));

    int* d_min_indices;
    cudaMalloc(&d_min_indices, points_num * 3 * sizeof(int));
    cudaMemset(d_min_indices, -1, points_num * 3 * sizeof(int));

    compute_intersect<<<(points_num + threadnum - 1) / threadnum, threadnum>>>(
        d_Ps, R, d_path_points, MAX_HEIGHT, points_num, path_num, path_len,  d_heights, d_normals, d_min_indices, d_dirs, false);

    torch::Tensor heights = torch::from_blob(d_heights, {points_num}, torch::kFloat).to(torch::kCUDA);
    heights = heights.clone();
    torch::Tensor normals = torch::from_blob(d_normals, {points_num, 3}, torch::kFloat).to(torch::kCUDA);
    normals = normals.clone();
    torch::Tensor dirs = torch::from_blob(d_dirs, {points_num, 3}, torch::kFloat).to(torch::kCUDA);
    dirs = dirs.clone();

    cudaFree(d_normals);
    cudaFree(d_min_indices);
    cudaFree(d_heights);
    cudaFree(d_dirs);
    if(!output_dirs) {
        // 如果不需要方向向量，则只返回高度和法向量
        return {heights, normals};
    }
    else{
        // 如果需要方向向量，则返回高度、法向量和方向向量
        return {heights, normals, dirs};
    }
}

// Python接口函数
std::vector<torch::Tensor> intersect_with_grad(torch::Tensor Ps, torch::Tensor path_points, torch::Tensor target_pos, float R, float MAX_HEIGHT, float n1, float n2, float z_plane_height)
{
    // Ps: 射线起点，形状 (N, 3)，[Px, Py, Pz]，固定常量
    // path_points: 路径点，形状 (path_num, path_len, 3)，每个路径点的 [x, y, z] 坐标
    // 需要预先使用 contiguous() 确保数据在连续内存中
    cudaSetDevice(Ps.get_device());
    int threadnum = 256;

    int points_num = Ps.size(0);
    int path_num = path_points.size(0);
    int path_len = path_points.size(1);

    float3* d_Ps = (float3*)Ps.data_ptr<ScalarType>();
    float3* d_path_points = (float3*)path_points.data_ptr<ScalarType>();
    float3* d_target_pos = (float3*)target_pos.data_ptr<ScalarType>();


    ScalarType* d_heights;
    cudaMalloc(&d_heights, points_num * sizeof(ScalarType));
    cudaMemset(d_heights, 0, points_num * sizeof(ScalarType));

    float3* d_normals;
    cudaMalloc(&d_normals, points_num * sizeof(float3));
    cudaMemset(d_normals, 0, points_num * sizeof(float3));

    int* d_min_indices;
    cudaMalloc(&d_min_indices, points_num * 3 * sizeof(int));
    cudaMemset(d_min_indices, -1, points_num * 3 * sizeof(int));

    ScalarType* d_loss;
    cudaMalloc(&d_loss, points_num * sizeof(ScalarType));
    cudaMemset(d_loss, 0, points_num * sizeof(ScalarType));

    ScalarType* d_grad;
    cudaMalloc(&d_grad, path_num * path_len * 3 * sizeof(ScalarType));
    cudaMemset(d_grad, 0, path_num * path_len * 3 * sizeof(ScalarType));

    ScalarType* d_grad_s_path_points;
    cudaMalloc(&d_grad_s_path_points, path_num * path_len * 3 * sizeof(ScalarType));
    cudaMemset(d_grad_s_path_points, 0, path_num * path_len * 3 * sizeof(ScalarType));

    ScalarType* d_grad_n_path_points;
    cudaMalloc(&d_grad_n_path_points, path_num * path_len * 3 * 3 * sizeof(ScalarType));
    cudaMemset(d_grad_n_path_points, 0, path_num * path_len * 3 * 3 * sizeof(ScalarType));

    float3* d_dirs;
    cudaMalloc(&d_dirs, points_num * sizeof(float3));
    cudaMemset(d_dirs, 0, points_num * sizeof(float3));

    compute_intersect<<<(points_num + threadnum - 1) / threadnum, threadnum>>>(
        d_Ps, R, d_path_points, MAX_HEIGHT, points_num, path_num, path_len,  d_heights, d_normals, d_min_indices, d_dirs, false);
    
    compute_gradients<<<(points_num + threadnum - 1) / threadnum, threadnum>>>(
        d_Ps, R, d_path_points, MAX_HEIGHT, points_num, path_num, path_len, d_target_pos, d_min_indices, z_plane_height, n1, n2,
    d_loss, d_grad, d_grad_s_path_points, d_grad_n_path_points);

    torch::Tensor heights = torch::from_blob(d_heights, {points_num}, torch::kFloat).to(torch::kCUDA);
    heights = heights.clone();
    torch::Tensor normals = torch::from_blob(d_normals, {points_num, 3}, torch::kFloat).to(torch::kCUDA);
    normals = normals.clone();
    torch::Tensor grad_s = torch::from_blob(d_grad_s_path_points, {path_num, path_len, 3}, torch::kFloat).to(torch::kCUDA);
    grad_s = grad_s.clone();
    torch::Tensor grad_n = torch::from_blob(d_grad_n_path_points, {path_num, path_len, 3, 3}, torch::kFloat).to(torch::kCUDA);
    grad_n = grad_n.clone();
    torch::Tensor loss = torch::from_blob(d_loss, {points_num}, torch::kFloat).to(torch::kCUDA);
    loss = loss.clone();
    torch::Tensor grad = torch::from_blob(d_grad, {path_num, path_len, 3}, torch::kFloat).to(torch::kCUDA);
    grad = grad.clone();

    cudaFree(d_normals);
    cudaFree(d_min_indices);
    cudaFree(d_heights);
    cudaFree(d_grad_s_path_points);
    cudaFree(d_grad_n_path_points);
    cudaFree(d_loss);
    cudaFree(d_grad);
    cudaFree(d_dirs);
    
    return {heights, normals, loss, grad};
}

// for height loss here:

// Python接口函数
std::vector<torch::Tensor> intersect_with_height_grad(torch::Tensor Ps, torch::Tensor path_points, float R, float MAX_HEIGHT, torch::Tensor target_height)
{
    // Ps: 射线起点，形状 (N, 3)，[Px, Py, Pz]，固定常量
    // path_points: 路径点，形状 (path_num, path_len, 3)，每个路径点的 [x, y, z] 坐标
    // 需要预先使用 contiguous() 确保数据在连续内存中
    cudaSetDevice(Ps.get_device());
    int threadnum = 256;

    int points_num = Ps.size(0);
    int path_num = path_points.size(0);
    int path_len = path_points.size(1);

    float3* d_Ps = (float3*)Ps.data_ptr<ScalarType>();
    float3* d_path_points = (float3*)path_points.data_ptr<ScalarType>();
    ScalarType* d_target_height = (ScalarType*)target_height.data_ptr<ScalarType>();


    ScalarType* d_heights;
    cudaMalloc(&d_heights, points_num * sizeof(ScalarType));
    cudaMemset(d_heights, 0, points_num * sizeof(ScalarType));

    ScalarType* d_heights_for_check;
    cudaMalloc(&d_heights_for_check, points_num * sizeof(ScalarType));
    cudaMemset(d_heights_for_check, 0, points_num * sizeof(ScalarType));

    float3* d_normals;
    cudaMalloc(&d_normals, points_num * sizeof(float3));
    cudaMemset(d_normals, 0, points_num * sizeof(float3));

    int* d_min_indices;
    cudaMalloc(&d_min_indices, points_num * 3 * sizeof(int));
    cudaMemset(d_min_indices, -1, points_num * 3 * sizeof(int));

    ScalarType* d_loss;
    cudaMalloc(&d_loss, points_num * sizeof(ScalarType));
    cudaMemset(d_loss, 0, points_num * sizeof(ScalarType));

    ScalarType* d_grad;
    cudaMalloc(&d_grad, path_num * path_len * 3 * sizeof(ScalarType));
    cudaMemset(d_grad, 0, path_num * path_len * 3 * sizeof(ScalarType));

    float3* d_dirs;
    cudaMalloc(&d_dirs, points_num * sizeof(float3));
    cudaMemset(d_dirs, 0, points_num * sizeof(float3));

    compute_intersect<<<(points_num + threadnum - 1) / threadnum, threadnum>>>(
        d_Ps, R, d_path_points, MAX_HEIGHT, points_num, path_num, path_len,  d_heights, d_normals, d_min_indices, d_dirs, true);
    
    compute_gradients_for_heights<<<(points_num + threadnum - 1) / threadnum, threadnum>>>(
        d_Ps, R, d_path_points, MAX_HEIGHT, points_num, path_num,  path_len, d_min_indices, d_target_height, d_loss, d_grad, d_heights_for_check, d_heights);


    torch::Tensor heights = torch::from_blob(d_heights, {points_num}, torch::kFloat).to(torch::kCUDA);
    heights = heights.clone();
    torch::Tensor normals = torch::from_blob(d_normals, {points_num, 3}, torch::kFloat).to(torch::kCUDA);
    normals = normals.clone();
    torch::Tensor loss = torch::from_blob(d_loss, {points_num}, torch::kFloat).to(torch::kCUDA);
    loss = loss.clone();
    torch::Tensor grad = torch::from_blob(d_grad, {path_num, path_len, 3}, torch::kFloat).to(torch::kCUDA);
    grad = grad.clone();
    torch::Tensor heights_for_check = torch::from_blob(d_heights_for_check, {points_num}, torch::kFloat).to(torch::kCUDA);
    heights_for_check = heights_for_check.clone();

    cudaFree(d_normals);
    cudaFree(d_min_indices);
    cudaFree(d_heights);
    cudaFree(d_loss);
    cudaFree(d_grad);
    cudaFree(d_dirs);
    cudaFree(d_heights_for_check);
    
    return {heights, normals, loss, grad, heights_for_check};
}


std::vector<torch::Tensor> intersect_with_normal_grad(torch::Tensor Ps, torch::Tensor path_points, torch::Tensor target_normals, float R, float MAX_HEIGHT, float n1, float n2, float z_plane_height)
{
    // Ps: 射线起点，形状 (N, 3)，[Px, Py, Pz]，固定常量
    // path_points: 路径点，形状 (path_num, path_len, 3)，每个路径点的 [x, y, z] 坐标
    // 需要预先使用 contiguous() 确保数据在连续内存中
    cudaSetDevice(Ps.get_device());
    int threadnum = 256;

    int points_num = Ps.size(0);
    int path_num = path_points.size(0);
    int path_len = path_points.size(1);

    float3* d_Ps = (float3*)Ps.data_ptr<ScalarType>();
    float3* d_path_points = (float3*)path_points.data_ptr<ScalarType>();
    float3* d_target_normals = (float3*)target_normals.data_ptr<ScalarType>();


    ScalarType* d_heights;
    cudaMalloc(&d_heights, points_num * sizeof(ScalarType));
    cudaMemset(d_heights, 0, points_num * sizeof(ScalarType));

    float3* d_normals;
    cudaMalloc(&d_normals, points_num * sizeof(float3));
    cudaMemset(d_normals, 0, points_num * sizeof(float3));

    int* d_min_indices;
    cudaMalloc(&d_min_indices, points_num * 3 * sizeof(int));
    cudaMemset(d_min_indices, -1, points_num * 3 * sizeof(int));

    ScalarType* d_loss;
    cudaMalloc(&d_loss, points_num * sizeof(ScalarType));
    cudaMemset(d_loss, 0, points_num * sizeof(ScalarType));

    ScalarType* d_grad;
    cudaMalloc(&d_grad, path_num * path_len * 3 * sizeof(ScalarType));
    cudaMemset(d_grad, 0, path_num * path_len * 3 * sizeof(ScalarType));

    float3* d_dirs;
    cudaMalloc(&d_dirs, points_num * sizeof(float3));
    cudaMemset(d_dirs, 0, points_num * sizeof(float3));

    compute_intersect<<<(points_num + threadnum - 1) / threadnum, threadnum>>>(
        d_Ps, R, d_path_points, MAX_HEIGHT, points_num, path_num, path_len,  d_heights, d_normals, d_min_indices, d_dirs, false);
    
    compute_gradients_for_normals<<<(points_num + threadnum - 1) / threadnum, threadnum>>>(
        d_Ps, R, d_path_points, MAX_HEIGHT, points_num, path_num, path_len, d_target_normals, d_min_indices, z_plane_height, n1, n2,
    d_loss, d_grad);

    torch::Tensor heights = torch::from_blob(d_heights, {points_num}, torch::kFloat).to(torch::kCUDA);
    heights = heights.clone();
    torch::Tensor normals = torch::from_blob(d_normals, {points_num, 3}, torch::kFloat).to(torch::kCUDA);
    normals = normals.clone();
    torch::Tensor loss = torch::from_blob(d_loss, {points_num}, torch::kFloat).to(torch::kCUDA);
    loss = loss.clone();
    torch::Tensor grad = torch::from_blob(d_grad, {path_num, path_len, 3}, torch::kFloat).to(torch::kCUDA);
    grad = grad.clone();

    cudaFree(d_normals);
    cudaFree(d_min_indices);
    cudaFree(d_heights);
    cudaFree(d_loss);
    cudaFree(d_grad);
    cudaFree(d_dirs);
    
    return {heights, normals, loss, grad};
}

// for renderer here

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
        Pi[0] = Jet(Pi_host.x); // 梯度索引 0
        Pi[1] = Jet(Pi_host.y, 1); // 梯度索引 1
        Pi[2] = Jet(Pi_host.z, 2); // 梯度索引 2
        Pj[0] = Jet(Pj_host.x); // 梯度索引 3
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
        
        /* // 累加 s 的梯度
        for (int i = 0; i < 3; ++i) {
            atomicAdd(&grad_s_path_points[base_idx * 3 + i], s_jet.grad[i]);
            atomicAdd(&grad_s_path_points[(base_idx+1) * 3 + i], s_jet.grad[i+3]);
        } */
        
    } else if(best_is_endpoint != -1){ // 端点球面交点
        int p_idx = best_path_idx * path_len + best_segment_idx;
        if (best_is_endpoint == 2) p_idx += 1; // 终点
        
        float3 P_end_host = d_path_points[p_idx];
        Jet P_end[3];
        P_end[0] = Jet(P_end_host.x); // 梯度索引 0
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
        
        /* // 累加 s 的梯度
        for (int i = 0; i < 3; ++i) {
            atomicAdd(&grad_s_path_points[p_idx * 3 + i], s_jet.grad[i]);
        } */
    }
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


    // 写入梯度
    if(best_is_endpoint == 0){
        // 管道
        int base_idx = best_path_idx * path_len + best_segment_idx;
        // 累加 loss 的梯度
        d_grad_render_to_path_points[base_idx * 2 + 0] += xj.grad[1] * d_grad_render_to_yj[idx * 2 + 0] + yj.grad[1] * d_grad_render_to_yj[idx * 2 + 1];
        d_grad_render_to_path_points[base_idx * 2 + 1] += xj.grad[2] * d_grad_render_to_yj[idx * 2 + 0] + yj.grad[2] * d_grad_render_to_yj[idx * 2 + 1];
        d_grad_render_to_path_points[(base_idx + 1) * 2 + 0] += xj.grad[4] * d_grad_render_to_yj[idx * 2 + 0] + yj.grad[4] * d_grad_render_to_yj[idx * 2 + 1];
        d_grad_render_to_path_points[(base_idx + 1) * 2 + 1] += xj.grad[5] * d_grad_render_to_yj[idx * 2 + 0] + yj.grad[5] * d_grad_render_to_yj[idx * 2 + 1];
    }   
    else if(best_is_endpoint > 0){
        // 端点
        int p_idx = best_path_idx * path_len + best_segment_idx;
        if (best_is_endpoint == 2) p_idx += 1; // 终点
        // 累加 loss 的梯度
        d_grad_render_to_path_points[p_idx * 2 + 0] += xj.grad[1] * d_grad_render_to_yj[idx * 2 + 0] + yj.grad[1] * d_grad_render_to_yj[idx * 2 + 1];
        d_grad_render_to_path_points[p_idx * 2 + 1] += xj.grad[2] * d_grad_render_to_yj[idx * 2 + 0] + yj.grad[2] * d_grad_render_to_yj[idx * 2 + 1];
    }
    
    
}

// Python接口函数
std::vector<torch::Tensor> intersect_with_grad_for_render(torch::Tensor Ps, torch::Tensor path_points, torch::Tensor grad_render_to_yj, float R, float MAX_HEIGHT, float n1, float n2, float z_plane_height)
{
    // Ps: 射线起点，形状 (N, 3)，[Px, Py, Pz]，固定常量
    // path_points: 路径点，形状 (path_num, path_len, 3)，每个路径点的 [x, y, z] 坐标
    // 需要预先使用 contiguous() 确保数据在连续内存中
    // grad_render_to_yj: 形状 (N, 2)，每个射线的梯度 [dL/dxj, dL/dyj]，其中 xj, yj 是投影点的坐标
    // 返回：grad_render_to_path_points：形状 (path_num, path_len, 2)，每个路径点的梯度 [dL/dyi, dL/dzi]，其中 yi, zi 是变量 i 的 y, z 坐标
    cudaSetDevice(Ps.get_device());
    int threadnum = 256;

    int points_num = Ps.size(0);
    int path_num = path_points.size(0);
    int path_len = path_points.size(1);

    float3* d_Ps = (float3*)Ps.data_ptr<ScalarType>();
    float3* d_path_points = (float3*)path_points.data_ptr<ScalarType>();
    ScalarType* d_grad_render_to_yj = grad_render_to_yj.data_ptr<ScalarType>(); // [N,2]

    ScalarType* d_heights;
    cudaMalloc(&d_heights, points_num * sizeof(ScalarType));
    cudaMemset(d_heights, 0, points_num * sizeof(ScalarType));

    float3* d_normals;
    cudaMalloc(&d_normals, points_num * sizeof(float3));
    cudaMemset(d_normals, 0, points_num * sizeof(float3));

    int* d_min_indices;
    cudaMalloc(&d_min_indices, points_num * 3 * sizeof(int));
    cudaMemset(d_min_indices, -1, points_num * 3 * sizeof(int));

    ScalarType* d_grad_render_to_path_points;
    cudaMalloc(&d_grad_render_to_path_points, path_num * path_len * 2 * sizeof(ScalarType));
    cudaMemset(d_grad_render_to_path_points, 0, path_num * path_len * 2 * sizeof(ScalarType));

    float3* d_dirs;
    cudaMalloc(&d_dirs, points_num * sizeof(float3));
    cudaMemset(d_dirs, 0, points_num * sizeof(float3));

    compute_intersect<<<(points_num + threadnum - 1) / threadnum, threadnum>>>(
        d_Ps, R, d_path_points, MAX_HEIGHT, points_num, path_num, path_len,  d_heights, d_normals, d_min_indices, d_dirs, false);
    
    compute_gradients_for_render<<<(points_num + threadnum - 1) / threadnum, threadnum>>>(
        d_Ps, R, d_path_points, MAX_HEIGHT, points_num, path_num, path_len, d_min_indices, z_plane_height, n1, n2,
        d_grad_render_to_yj, d_grad_render_to_path_points);

    torch::Tensor heights = torch::from_blob(d_heights, {points_num}, torch::kFloat).to(torch::kCUDA);
    heights = heights.clone();
    torch::Tensor normals = torch::from_blob(d_normals, {points_num, 3}, torch::kFloat).to(torch::kCUDA);
    normals = normals.clone();
    torch::Tensor grad_render_to_path_points = torch::from_blob(d_grad_render_to_path_points, {path_num, path_len, 2}, torch::kFloat).to(torch::kCUDA);
    grad_render_to_path_points = grad_render_to_path_points.clone();

    cudaFree(d_normals);
    cudaFree(d_min_indices);
    cudaFree(d_heights);
    cudaFree(d_grad_render_to_path_points);
    cudaFree(d_dirs);
    
    return {grad_render_to_path_points};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("intersect", intersect);
    m.def("intersect_with_grad", intersect_with_grad);
    m.def("intersect_with_grad_for_render", intersect_with_grad_for_render);
    m.def("intersect_with_height_grad", intersect_with_height_grad);
    m.def("intersect_with_normal_grad", intersect_with_normal_grad);
}
