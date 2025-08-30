#pragma once

#include <cuda_runtime.h>
#include <math.h>

// 使用CUDA内置的float3类型，不再重定义
// CUDA已经在vector_types.h中定义了float3结构体

// 为float3添加向量操作符
__device__ __host__ inline float3 operator-(const float3& a, const float3& b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ __host__ inline float3 operator+(const float3& a, const float3& b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ __host__ inline float3 operator*(const float3& a, float s) {
    return make_float3(a.x * s, a.y * s, a.z * s);
}

__device__ __host__ inline float3 operator*(float s, const float3& a) {
    return make_float3(a.x * s, a.y * s, a.z * s);
}

__device__ __host__ inline float3 operator/(const float3& a, float s) {
    return make_float3(a.x / s, a.y / s, a.z / s);
}

__device__ __host__ inline float dot(const float3& a, const float3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ __host__ inline float length(const float3& a) {
    return sqrtf(dot(a, a));
}

__device__ __host__ inline float3 normalize(const float3& a) {
    float len = length(a);
    if (len > 1e-6f) {
        return a / len;
    }
    return make_float3(0.0f, 0.0f, 1.0f);
}

// CUDA自带的helper函数
// 如果您已经包含了helper_math.h，那么可以移除这些自定义的操作符
// CUDA提供的helper_math.h头文件中已经定义了这些操作符
// 您可以检查是否需要包含helper_math.h代替自己定义这些操作符
