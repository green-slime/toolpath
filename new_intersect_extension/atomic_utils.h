#pragma once

#include <torch/extension.h>

// 保留原子操作的兼容性代码

#if !defined (__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
// 新架构有原生atomicAdd(double)
#else
__device__ inline double atomicAdd(double* address, double val)
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

__device__ inline float atomicMinFloat(float* addr, float value) {
    // 快速路径：先检查是否需要更新
    float current = *addr;
    if (current <= value || isnan(value)) return current;
    
    // 慢速路径：原子更新
    int* addr_as_int = (int*)addr;
    int old = *addr_as_int;
    int assumed;
    
    do {
        assumed = old;
        float old_val = __int_as_float(assumed);
        if (isnan(old_val) || old_val <= value) break;
        old = atomicCAS(addr_as_int, assumed, __float_as_int(value));
    } while (assumed != old);
    
    return __int_as_float(old);
}

// 在 atomic_utils.h 中添加
struct PackedIntersectData {
    union {
        struct {
            float height;           // 4字节
            uint16_t path_idx;      // 2字节  
            uint16_t seg_idx;       // 2字节
        };
        uint64_t packed;           // 8字节总计
    };
    
    __device__ PackedIntersectData() : height(1e30f), path_idx(0xFFFF), seg_idx(0xFFFF) {}
    __device__ PackedIntersectData(float h, uint16_t p, uint16_t s) : height(h), path_idx(p), seg_idx(s) {}
};

// 原子更新函数
__device__ inline bool atomicMinPackedData(uint64_t* addr, float new_height, uint16_t path_idx, uint16_t seg_idx) {
    PackedIntersectData new_data(new_height, path_idx, seg_idx);
    
    uint64_t old = *addr;
    uint64_t assumed;
    
    do {
        assumed = old;
        PackedIntersectData old_data;
        old_data.packed = assumed;
        
        // 如果新高度不更小，不更新
        if (old_data.height <= new_height) return false;
        
        old = atomicCAS((unsigned long long*)addr, assumed, new_data.packed);
    } while (assumed != old);
    
    return true;  // 成功更新
}

