#ifndef JET_H
#define JET_H

#include <math.h>
#include <cuda_runtime.h>

struct Jet {
    float val;      // 值
    float grad[6];  // 梯度（最多6个变量：两个点的xyz）

    __device__ Jet(float v = 0, int index = -1) : val(v) {
        for (int i = 0; i < 6; ++i) grad[i] = 0;
        if (index >= 0 && index < 6) grad[index] = 1;
    }

    __device__ Jet operator+(const Jet& other) const {
        Jet res(val + other.val);
        for (int i = 0; i < 6; ++i) 
            res.grad[i] = grad[i] + other.grad[i];
        return res;
    }

    __device__ Jet operator-(const Jet& other) const {
        Jet res(val - other.val);
        for (int i = 0; i < 6; ++i) 
            res.grad[i] = grad[i] - other.grad[i];
        return res;
    }

    __device__ Jet operator*(const Jet& other) const {
        Jet res(val * other.val);
        for (int i = 0; i < 6; ++i) 
            res.grad[i] = grad[i] * other.val + val * other.grad[i];
        return res;
    }

    /* __device__ Jet operator/(const Jet& other) const {
        float inv = 1.0f / other.val;
        float inv2 = inv * inv;
        Jet res(val * inv);
        for (int i = 0; i < 6; ++i)
            res.grad[i] = (grad[i] * other.val - val * other.grad[i]) * inv2;
        return res;
    } */

    __device__ Jet operator/(const Jet& other) const{
        Jet res(val / other.val);
        for (int i=0; i < 6; ++i)
            res.grad[i] = (grad[i] * other.val - val * other.grad[i]) / (other.val * other.val);
        return res;
    }

    __device__ Jet operator-() const {
        Jet res(-val);
        for (int i = 0; i < 6; ++i) 
            res.grad[i] = -grad[i];
        return res;
    }
};

// 三维向量 Jet 版本
struct Jet3 {
    Jet x, y, z;

    __device__ Jet3(Jet x_ = Jet(), Jet y_ = Jet(), Jet z_ = Jet())
        : x(x_), y(y_), z(z_) {}
    
    __device__ Jet3(float3 v) : x(v.x), y(v.y), z(v.z) {}
    
    __device__ Jet3 operator+(const Jet3& other) const {
        return Jet3(x + other.x, y + other.y, z + other.z);
    }
    
    __device__ Jet3 operator-(const Jet3& other) const {
        return Jet3(x - other.x, y - other.y, z - other.z);
    }
    
    __device__ Jet3 operator*(const Jet& scalar) const {
        return Jet3(x * scalar, y * scalar, z * scalar);
    }

    // Jet * Jet3 (声明为友元或独立函数)
    friend __device__ Jet3 operator*(const Jet& scalar, const Jet3& vec) {
        return Jet3(scalar * vec.x, scalar * vec.y, scalar * vec.z);
    }
    
    __device__ Jet3 operator/(const Jet& scalar) const {
        return Jet3(x / scalar, y / scalar, z / scalar);
    }
};

// Jet3 点积
__device__ Jet dot(const Jet3& a, const Jet3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ Jet norm2(const Jet3& a){
    return dot(a,a);
}

// Jet版开方
__device__ Jet sqrt(const Jet& a) {
    float root = sqrtf(a.val);
    Jet res(root);
    for (int i = 0; i < 6; ++i)
        res.grad[i] = 0.5f / root * a.grad[i];
    return res;
}

// Jet3 归一化
__device__ Jet3 normalize(const Jet3& v) {
    Jet len = sqrt(dot(v, v));
    return Jet3(v.x / len, v.y / len, v.z / len);
}


// Jet版平方
__device__ Jet pow2(const Jet& a) {
    Jet res(a.val * a.val);
    for (int i = 0; i < 6; ++i)
        res.grad[i] = 2.0f * a.val * a.grad[i];
    return res;
}

#endif // JET_H