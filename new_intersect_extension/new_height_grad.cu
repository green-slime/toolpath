#include <torch/extension.h>
#include <iostream>
#include <vector>
#include "geometry_utils.h"
#include "Jet.h"
#include "all_kernels.h"
#include "atomic_utils.h"

using namespace std;

__global__ void compute_gradients_for_heights(
    float3* d_Ps, float R, float3* d_path_points, float MAX_HEIGHT,
    int points_num, int path_num, int path_len, 
    int* min_indices, float gouge_weight,
    ScalarType* d_target_height, // 目标高度
    ScalarType* d_loss, // 输出每个光束产生的 loss
    ScalarType* d_grad // 输出 loss 之和对各变量的梯度
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
        P_end[0] = Jet(P_end_host.x, 0); // 梯度索引 0
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
        loss = Jet(gouge_weight) * (Jet(target_height) - height_jet) * (Jet(target_height) - height_jet); // 使用平方损失
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

std::vector<torch::Tensor> intersect_with_height_grad(
    torch::Tensor Ps, torch::Tensor path_points, float R, 
    float MAX_HEIGHT, torch::Tensor target_height, float gouge_weight)
{
    // 这里输入[Nx, Ny, 3]的Ps
    // path_points [path_num, path_len, 3]
    cudaSetDevice(Ps.get_device());

    int Nx = Ps.size(0);
    int Ny = Ps.size(1);
    int points_num = Nx * Ny;
    int path_num = path_points.size(0);
    int path_len = path_points.size(1);
    int total_segments = path_num * (path_len - 1);
    
    // ===== 修改1: 使用打包数据结构 =====
    // 分配打包数据内存
    auto packed_data = torch::zeros({points_num}, 
        torch::TensorOptions().dtype(torch::kInt64).device(Ps.device()));
    // 初始化为最大高度
    
    auto normals = torch::full({points_num, 3}, 0.0f, 
                               torch::TensorOptions().dtype(torch::kFloat32).device(Ps.device()));
    normals.select(1, 2).fill_(1.0f);  // 设置z分量为1，即(0,0,1)
    auto dirs = torch::full({points_num, 3}, 0.0f, 
                           torch::TensorOptions().dtype(torch::kFloat32).device(Ps.device()));
    auto loss = torch::zeros({points_num}, 
                            torch::TensorOptions().dtype(torch::kFloat32).device(Ps.device()));
    auto grad = torch::zeros({path_num, path_len, 3}, 
                            torch::TensorOptions().dtype(torch::kFloat32).device(Ps.device()));

    // 获取数据指针
    float3* d_Ps = reinterpret_cast<float3*>(Ps.data_ptr<float>());
    float3* d_path_points = reinterpret_cast<float3*>(path_points.data_ptr<float>());
    uint64_t* d_packed_data = reinterpret_cast<uint64_t*>(packed_data.data_ptr<int64_t>());
    float3* d_normals = reinterpret_cast<float3*>(normals.data_ptr<float>());
    float3* d_dirs = reinterpret_cast<float3*>(dirs.data_ptr<float>());
    float* d_loss = loss.data_ptr<float>();
    float* d_grad = grad.data_ptr<float>();
    float* d_target_height = target_height.data_ptr<float>();

    // 初始化打包数据（在GPU上）
    int threadsPerBlock = 256;
    int blocksPerGrid_init = (points_num + threadsPerBlock - 1) / threadsPerBlock;
    
    // 添加初始化核函数
    init_packed_data<<<blocksPerGrid_init, threadsPerBlock>>>(
        d_packed_data, points_num, MAX_HEIGHT
    );

    // ===== 修改2: 调用新的intersect函数 =====
    int blocksPerGrid_segments = (total_segments + threadsPerBlock - 1) / threadsPerBlock;
    
    compute_intersect_object_parallel_v2<<<blocksPerGrid_segments, threadsPerBlock>>>(
        d_Ps, R, d_path_points, MAX_HEIGHT, points_num, path_num, path_len, Nx, Ny,
        d_packed_data, d_normals, d_dirs  // 使用打包数据
    );

    // ===== 修改3: 解包数据并创建min_indices =====
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

    // ===== 修改4: 调用梯度计算核函数 =====
    compute_gradients_for_heights<<<blocksPerGrid_points, threadsPerBlock>>>(
        d_Ps, R, d_path_points, MAX_HEIGHT, points_num, path_num, path_len,
        d_min_indices, gouge_weight, 
        d_target_height, d_loss, d_grad
    );

    cudaDeviceSynchronize();

    return {heights, 
            normals, 
            loss, 
            grad};
}