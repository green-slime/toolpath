#pragma once

#include <torch/extension.h>
#include <iostream>
#include <vector>
#include "geometry_utils.h"
#include "Jet.h"

typedef float ScalarType;

// new_intersect.cu:

__global__ void compute_intersect_object_parallel_v2(
    float3* d_Ps, ScalarType R, float3* d_path_points, ScalarType MAX_HEIGHT, 
    int points_num, int path_num, int path_len, int Nx, int Ny, 
    uint64_t* packed_data,
    float3* normals, float3* dirs);

std::vector<torch::Tensor> intersect_object_parallel_v2(
    torch::Tensor Ps, torch::Tensor path_points, 
    float R, float MAX_HEIGHT);

__global__ void unpack_intersect_data(
    uint64_t* packed_data, int points_num,
    float* heights, int* min_indices);

__global__ void compute_intersect_details(
    float3* d_Ps, ScalarType R, float3* d_path_points, 
    int points_num, int path_len, float MAX_HEIGHT,
    int* min_indices, float* heights, float3* normals, float3* dirs);

__global__ void init_packed_data(uint64_t* packed_data, int points_num, float MAX_HEIGHT);

// new_height_grad.cu:

std::vector<torch::Tensor> intersect_with_height_grad(
    torch::Tensor Ps, torch::Tensor path_points, float R, float MAX_HEIGHT, torch::Tensor target_height, float gouge_weight = 5.0f);

// new_render_grad.cu
std::vector<torch::Tensor> intersect_with_grad_for_render(
    torch::Tensor Ps, torch::Tensor path_points, torch::Tensor grad_render_to_yj, 
    float R, float MAX_HEIGHT, float n1, float n2, float z_plane_height);


