#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <iostream>
#include <vector>
#include "clip.h"

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

// 核心渲染kernel
__global__ void compute_render(ScalarType* d_compute_tri_area, int* d_faces, 
    ScalarType* d_proj_xy, ScalarType* d_total_gray_value, ScalarType* d_render_result, int* d_face_num, ScalarType* d_edge_len, int* d_img_size)
{
    // d_proj_xy 是实际xy坐标
    // tri_coor 是在640*640的网格上，640为img_size，可变
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < d_face_num[0])
    {
        ScalarType gray_value = d_compute_tri_area[tid] * d_total_gray_value[0];

        int point_idx[3] = { d_faces[3*tid], d_faces[3*tid+1], d_faces[3*tid+2]};
        ScalarType tri_coor[6];
        int poly_num = 3;
        int img_size = d_img_size[0];
        ScalarType max_edge_len=d_edge_len[0];
        for(int i = 0; i < 3; i++)
        {
            tri_coor[2*i] = d_proj_xy[2*point_idx[i]]*img_size/max_edge_len;
            tri_coor[2*i+1] = d_proj_xy[2*point_idx[i]+1]*img_size/max_edge_len;
        }
        ScalarType big_area = compute_area(tri_coor, 3);
        
        // 计算包围盒
        ScalarType max_x = tri_coor[0], max_y = tri_coor[1], min_x = tri_coor[0], min_y = tri_coor[1];
        for(int i = 0; i < 3; i++)
        {
            if(tri_coor[2*i] < min_x) min_x = tri_coor[2*i];
            if(tri_coor[2*i] > max_x) max_x = tri_coor[2*i];
            if(tri_coor[2*i+1] < min_y) min_y = tri_coor[2*i+1];
            if(tri_coor[2*i+1] > max_y) max_y = tri_coor[2*i+1];
        }

        int low_bound_x = (int)min_x;
        int low_bound_y = (int)min_y;
        int high_bound_x = (int)max_x + 1;
        int high_bound_y = (int)max_y + 1;
        
        low_bound_x = (low_bound_x > 0) ? low_bound_x : 0;
        low_bound_y = (low_bound_y > 0) ? low_bound_y : 0;
        high_bound_x = (high_bound_x < img_size) ? high_bound_x : img_size;
        high_bound_y = (high_bound_y < img_size) ? high_bound_y : img_size;

        ScalarType clipsquare[4];
        ScalarType new_poly[max_edge_num];
        int new_poly_num;

        // 光栅化
        for(int i_x = low_bound_x; i_x < high_bound_x; i_x++)
        {
            for(int i_y = low_bound_y; i_y < high_bound_y; i_y++)
            {
                int grid = i_x * img_size + i_y; 
                //int grid = (639-i_y) * img_size + i_x;
                clipsquare[0] = (ScalarType)i_x;
                clipsquare[1] = (ScalarType)i_y;
                clipsquare[2] = (ScalarType)(i_x + 1);
                clipsquare[3] = (ScalarType)(i_y + 1);

                clip_Polygon(tri_coor, poly_num, clipsquare, new_poly, new_poly_num);
                ScalarType area = compute_area(new_poly, new_poly_num);

                if(big_area != 0)
                {
                    ScalarType local_contribution = gray_value * area / big_area;
                    atomicAdd(&d_render_result[grid], local_contribution);
                }  
            }
        }
    }
}

__global__ void compute_diff(ScalarType* d_compute_tri_area, int* d_faces, 
    ScalarType* d_proj_xy, ScalarType* d_total_gray_value, ScalarType* d_render_result, ScalarType* d_real_picture, ScalarType* d_render_diff,
    ScalarType* d_points_diff, ScalarType* d_tri_area_diff, int* d_face_num, ScalarType* d_sobel_weight, ScalarType* d_points_sobel_diff, ScalarType* d_edge_len, int* d_img_size, ScalarType* d_points_diff_l1)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < d_face_num[0])
    {
        ScalarType gray_value = d_compute_tri_area[tid] * d_total_gray_value[0];

        int point_idx[3] = { d_faces[3*tid], d_faces[3*tid+1], d_faces[3*tid+2]};
        ScalarType tri_coor[6];
        ScalarType x_diff[3],y_diff[3];
        int poly_num = 3;
        int img_size = d_img_size[0];
        ScalarType max_edge_len=d_edge_len[0];
        for(int i = 0; i < 3; i++)
        {
            tri_coor[2*i] = d_proj_xy[2*point_idx[i]]*img_size/max_edge_len;
            tri_coor[2*i+1] = d_proj_xy[2*point_idx[i]+1]*img_size/max_edge_len;
        }

        // 这一部分是取 tri_coor 三个坐标之间的距离进行判断
        // 以免出现 Δx 或 Δy 过小或过大的情况
        for(int i = 0; i < 3; i++)
        {
            x_diff[i] = tri_coor[2*((i+1)%3)] - tri_coor[2*i];
            if(x_diff[i]<0)
            {
                x_diff[i] = -x_diff[i];
            }

            y_diff[i] = tri_coor[2*((i+1)%3)+1] - tri_coor[2*i+1];
            if(y_diff[i]<0)
            {
                y_diff[i] = -y_diff[i];
            }
        }

        ScalarType x_step = x_diff[0],y_step = y_diff[0];
        for(int i = 1;i<3;i++)
        {
            if(x_diff[i]<x_step)
            {
                x_step = x_diff[i];
            }
            if(y_diff[i]<y_step)
            {
                y_step = y_diff[i];
            }
        }
        x_step = x_step*0.01;
        y_step = y_step*0.01;
        float min_step = 1e-5;
        if(x_step < min_step)
        {
            x_step = min_step;
        }
        if(y_step < min_step)
        {
            y_step = min_step;
        }

        ScalarType big_area = compute_area( tri_coor, 3);
        ScalarType new_big_area[6];
        for(int i=0;i<6;i++)
        {
            if(i%2 == 0)
            {
                tri_coor[i] = tri_coor[i] + x_step;
                new_big_area[i] = compute_area( tri_coor, 3);
                tri_coor[i] = tri_coor[i] - x_step;
            }
            else
            {
                tri_coor[i] = tri_coor[i] + y_step;
                new_big_area[i] = compute_area( tri_coor, 3);
                tri_coor[i] = tri_coor[i] - y_step;
            }
        }

        ScalarType max_x = tri_coor[0], max_y = tri_coor[1], min_x = tri_coor[0], min_y=tri_coor[1];
        for(int i=0;i<3;i++)
        {
            if(tri_coor[2*i]<min_x)
            {
                min_x = tri_coor[2*i];
            }
            if(tri_coor[2*i]>max_x)
            {
                max_x = tri_coor[2*i];
            }
            if(tri_coor[2*i+1]<min_y)
            {
                min_y = tri_coor[2*i+1];
            }
            if(tri_coor[2*i+1]>max_y)
            {
                max_y = tri_coor[2*i+1];
            }
        }
        int low_bound_x=(int)((min_x));
        int low_bound_y=(int)((min_y));
        int high_bound_x=(int)((max_x))+1;
        int high_bound_y=(int)((max_y))+1;
        
        low_bound_x=(low_bound_x>0)?low_bound_x:0;
        low_bound_y=(low_bound_y>0)?low_bound_y:0;
        high_bound_x=(high_bound_x<img_size)?high_bound_x:img_size;
        high_bound_y=(high_bound_y<img_size)?high_bound_y:img_size;
        ScalarType clipsquare[4];
        ScalarType new_poly[max_edge_num];
        int new_poly_num;
        ScalarType local_area_diff = 0, local_sobel_diff = 0;

        for(int i_x = low_bound_x; i_x < high_bound_x; i_x++)
        {
            for(int i_y = low_bound_y; i_y < high_bound_y; i_y++)
            {
                int grid = i_x * img_size + i_y; 
                //int grid = (img_size-1-i_y) * img_size + i_x;
                clipsquare[0] = (ScalarType)i_x;
                clipsquare[1] = (ScalarType)i_y;
                clipsquare[2] = (ScalarType)(i_x + 1);
                clipsquare[3] = (ScalarType)(i_y + 1);

                clip_Polygon( tri_coor, poly_num, clipsquare, new_poly, new_poly_num);
                ScalarType area = compute_area( new_poly, new_poly_num);
                ScalarType local_contribution;
                d_render_diff[grid] = d_render_result[grid] - d_real_picture[grid];
                if(big_area!=0)
                {
                    local_contribution = gray_value*area/big_area;
                    local_area_diff = local_area_diff + d_render_diff[grid]*area/big_area;
                    local_sobel_diff = local_sobel_diff + d_sobel_weight[grid]*area/big_area;
                }  

                for(int i = 0;i < 6;i++)
                {
                    if(i%2 == 0)
                    {
                        tri_coor[i] = tri_coor[i] + x_step;
                        clip_Polygon( tri_coor, poly_num, clipsquare, new_poly, new_poly_num);
                        ScalarType area = compute_area( new_poly, new_poly_num);
                        ScalarType new_local_contribution;
                        if(new_big_area[i]!=0)
                        {
                            new_local_contribution = gray_value*area/new_big_area[i];
                            ScalarType local_point_diff = 2 * d_render_diff[grid] * ( new_local_contribution - local_contribution)/x_step*img_size/max_edge_len;
                            atomicAdd(&d_points_diff[2*point_idx[i/2]], local_point_diff);

                            ScalarType local_point_diff_l1 = (d_render_diff[grid] >= 0 ? 1.0 : -1.0) * (new_local_contribution - local_contribution) / x_step * img_size / max_edge_len;
                            atomicAdd(&d_points_diff_l1[2*point_idx[i/2]], local_point_diff_l1);
            
                            local_point_diff = 2 * d_sobel_weight[grid] * ( new_local_contribution - local_contribution)/x_step*img_size/max_edge_len;
                            atomicAdd(&d_points_sobel_diff[2*point_idx[i/2]], local_point_diff);
                        }
                        tri_coor[i] = tri_coor[i] - x_step;
                    }
                    else
                    {
                        tri_coor[i] = tri_coor[i] + y_step;
                        clip_Polygon( tri_coor, poly_num, clipsquare, new_poly, new_poly_num);
                        ScalarType area = compute_area( new_poly, new_poly_num);
                        ScalarType new_local_contribution;
                        if(new_big_area[i]!=0)
                        {
                            new_local_contribution = gray_value*area/new_big_area[i];
                            ScalarType local_point_diff = 2 * d_render_diff[grid] * ( new_local_contribution - local_contribution)/y_step*img_size/max_edge_len;
                            atomicAdd(&d_points_diff[2*point_idx[(i-1)/2]+1] , local_point_diff);

                            ScalarType local_point_diff_l1 = (d_render_diff[grid] >= 0 ? 1.0 : -1.0) * (new_local_contribution - local_contribution) / y_step * img_size / max_edge_len;
                            atomicAdd(&d_points_diff_l1[2*point_idx[(i-1)/2]+1], local_point_diff_l1);

                            local_point_diff = 2 * d_sobel_weight[grid] * ( new_local_contribution - local_contribution)/y_step*img_size/max_edge_len;
                            atomicAdd(&d_points_sobel_diff[2*point_idx[(i-1)/2]+1] , local_point_diff);
                        }
                        tri_coor[i] = tri_coor[i] - y_step;
                    }
                }
            }
        }
        local_area_diff = local_area_diff*d_total_gray_value[0];
        atomicAdd(&d_tri_area_diff[tid] , local_area_diff);
    }
}

__global__ void compute_sobel( ScalarType* d_picture_sobel_x, ScalarType* d_picture_sobel_y, ScalarType* d_real_picture, int* d_gpudata_size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < (int)(d_gpudata_size[0]*d_gpudata_size[0]))
    {
        int data_size = (int)d_gpudata_size[0];
        int i_x = tid / data_size;
        int i_y = tid % data_size;
        /* if(i_x>=1&&i_x<(data_size-1)&&i_y>=1&&i_y<(data_size-1))
        {
            d_picture_sobel_x[tid] = (d_real_picture[tid + data_size] - d_real_picture[tid - data_size]);
            d_picture_sobel_y[tid] = (d_real_picture[tid + 1] - d_real_picture[tid - 1]);
        }
        else {
            d_picture_sobel_x[tid] = 0;
            d_picture_sobel_y[tid] = 0;
        } */
       if(i_x==0){
        d_picture_sobel_x[tid] = (d_real_picture[tid + data_size] - 0);
       }
       else if(i_x==data_size-1){
        d_picture_sobel_x[tid] = (0 - d_real_picture[tid - data_size]);
       }
       else d_picture_sobel_x[tid] = (d_real_picture[tid + data_size] - d_real_picture[tid - data_size]);
       if(i_y==0){
        d_picture_sobel_y[tid] = (d_real_picture[tid + 1] - 0);
       }
       else if(i_y==data_size-1){
        d_picture_sobel_y[tid] = (0 - d_real_picture[tid - 1]);
       }
       else d_picture_sobel_y[tid] = (d_real_picture[tid + 1] - d_real_picture[tid - 1]);

    }
}

__global__ void compute_sub( ScalarType* a, ScalarType* b, ScalarType* c, int* d_gpudata_size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < (int)(d_gpudata_size[0]*d_gpudata_size[0]))
    {
        c[tid] = a[tid] - b[tid];
    }
}

__global__ void compute_sobel_weight( ScalarType* d_sobel_x_diff, ScalarType* d_sobel_y_diff, ScalarType* d_sobel_weight, int* d_gpudata_size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < (int)(d_gpudata_size[0]*d_gpudata_size[0]))
    {
        int data_size = (int)d_gpudata_size[0];
        int x = tid / data_size;
        int y = tid % data_size;

        if(x<=data_size-2&&y<=data_size-1&&y>=0)
        {
            d_sobel_weight[tid] = d_sobel_weight[tid] - d_sobel_x_diff[tid+data_size];
        }

        if(x>=1&&y<=data_size-1&&y>=0)
        {
            d_sobel_weight[tid] = d_sobel_weight[tid] + d_sobel_x_diff[tid-data_size];
        }

        if(y<=data_size-2&&x<=data_size-1&&x>=0)
        {
            d_sobel_weight[tid] = d_sobel_weight[tid] - d_sobel_y_diff[tid+1];
        }

        if(y>=1&&x<=data_size-1&&x>=0)
        {
            d_sobel_weight[tid] = d_sobel_weight[tid] + d_sobel_y_diff[tid-1];
        }

        /* if(x<data_size-2&&y<data_size-1&&y>0)
        {
            d_sobel_weight[tid] = d_sobel_weight[tid] - d_sobel_x_diff[tid+data_size];
        }

        if(x>1&&y<data_size-1&&y>0)
        {
            d_sobel_weight[tid] = d_sobel_weight[tid] + d_sobel_x_diff[tid-data_size];
        }

        if(y<data_size-2&&x<data_size-1&&x>0)
        {
            d_sobel_weight[tid] = d_sobel_weight[tid] - d_sobel_y_diff[tid+1];
        }

        if(y>1&&x<data_size-1&&x>0)
        {
            d_sobel_weight[tid] = d_sobel_weight[tid] + d_sobel_y_diff[tid-1];
        } */

    }
}

// Python接口函数
std::vector<torch::Tensor> render(torch::Tensor compute_tri_area, torch::Tensor faces, 
    torch::Tensor proj_xy, torch::Tensor total_gray_value, torch::Tensor edge_len, int img_size)
{
    cudaSetDevice(compute_tri_area.get_device());
    int threadnum = 256;

    ScalarType* d_compute_tri_area = compute_tri_area.data_ptr<ScalarType>();
    int* d_faces = faces.data_ptr<int>();
    ScalarType* d_proj_xy = proj_xy.data_ptr<ScalarType>();
    ScalarType* d_total_gray_value = total_gray_value.data_ptr<ScalarType>();
    ScalarType* d_edge_len = edge_len.data_ptr<ScalarType>();

    ScalarType* d_render_result;
    cudaMalloc(&d_render_result, img_size * img_size * sizeof(ScalarType));
    cudaMemset(d_render_result, 0, img_size * img_size * sizeof(ScalarType));

    int face_num = faces.size(0);
    int* d_face_num;
    cudaMalloc(&d_face_num, sizeof(int));
    cudaMemcpy(d_face_num, &face_num, sizeof(int), cudaMemcpyHostToDevice);
    int* d_img_size;
    cudaMalloc(&d_img_size, sizeof(int));
    cudaMemcpy(d_img_size, &img_size, sizeof(int), cudaMemcpyHostToDevice);

    compute_render<<<(face_num + threadnum - 1) / threadnum, threadnum>>>(
        d_compute_tri_area, d_faces, d_proj_xy, 
        d_total_gray_value, d_render_result, d_face_num, d_edge_len, d_img_size);

    torch::Tensor render_result = torch::from_blob(d_render_result, {img_size, img_size}, torch::kFloat).to(torch::kCUDA);
    render_result = render_result.clone();

    cudaFree(d_face_num);
    cudaFree(d_img_size);
    cudaFree(d_render_result);
    
    return {render_result};
}

std::vector<torch::Tensor> diff( torch::Tensor compute_tri_area, torch::Tensor faces, torch::Tensor proj_xy, torch::Tensor total_gray_value, torch::Tensor real_picture, torch::Tensor real_sobel_x, torch::Tensor real_sobel_y, torch::Tensor edge_len, int img_size)
{
    cudaSetDevice(compute_tri_area.get_device());
    int threadnum = 256;

    ScalarType* d_compute_tri_area = compute_tri_area.data_ptr<ScalarType>();
    int* d_faces = faces.data_ptr<int>();
    ScalarType* d_proj_xy = proj_xy.data_ptr<ScalarType>();
    ScalarType* d_total_gray_value = total_gray_value.data_ptr<ScalarType>();
    ScalarType* d_real_picture = real_picture.data_ptr<ScalarType>();
    ScalarType* d_real_sobel_x = real_sobel_x.data_ptr<ScalarType>();
    ScalarType* d_real_sobel_y = real_sobel_y.data_ptr<ScalarType>();
    ScalarType* d_edge_len = edge_len.data_ptr<ScalarType>();

    ScalarType* d_render_result;
    cudaMalloc(&d_render_result, img_size * img_size * sizeof(ScalarType));
    cudaMemset(d_render_result, 0, img_size * img_size * sizeof(ScalarType));

    ScalarType* d_render_diff;
    cudaMalloc(&d_render_diff, img_size * img_size * sizeof(ScalarType));
    cudaMemset(d_render_diff, 0, img_size * img_size * sizeof(ScalarType));

    ScalarType* d_render_sobel_x;
    cudaMalloc(&d_render_sobel_x, img_size * img_size * sizeof(ScalarType));
    cudaMemset(d_render_sobel_x, 0, img_size * img_size * sizeof(ScalarType));

    ScalarType* d_render_sobel_y;
    cudaMalloc(&d_render_sobel_y, img_size * img_size * sizeof(ScalarType));
    cudaMemset(d_render_sobel_y, 0, img_size * img_size * sizeof(ScalarType));

    ScalarType* d_sobel_x_diff;
    cudaMalloc(&d_sobel_x_diff, img_size * img_size * sizeof(ScalarType));
    cudaMemset(d_sobel_x_diff, 0, img_size * img_size * sizeof(ScalarType));

    ScalarType* d_sobel_y_diff;
    cudaMalloc(&d_sobel_y_diff, img_size * img_size * sizeof(ScalarType));
    cudaMemset(d_sobel_y_diff, 0, img_size * img_size * sizeof(ScalarType));

    ScalarType* d_sobel_weight;
    cudaMalloc(&d_sobel_weight, img_size * img_size * sizeof(ScalarType));
    cudaMemset(d_sobel_weight, 0, img_size * img_size * sizeof(ScalarType));

    int points_num = proj_xy.numel()/2;
    ScalarType* d_points_diff;
    cudaMalloc(&d_points_diff, 2 * points_num * sizeof(ScalarType));
    cudaMemset(d_points_diff, 0, 2 * points_num * sizeof(ScalarType));

    ScalarType* d_points_diff_l1;
    cudaMalloc(&d_points_diff_l1, 2 * points_num * sizeof(ScalarType));
    cudaMemset(d_points_diff_l1, 0, 2 * points_num * sizeof(ScalarType));

    ScalarType* d_points_sobel_diff;
    cudaMalloc(&d_points_sobel_diff, 2 * points_num * sizeof(ScalarType));
    cudaMemset(d_points_sobel_diff, 0, 2 * points_num * sizeof(ScalarType));

    int origin_faces_num = compute_tri_area.numel();
    ScalarType* d_tri_area_diff;
    cudaMalloc(&d_tri_area_diff, origin_faces_num * sizeof(ScalarType));
    cudaMemset(d_tri_area_diff, 0, origin_faces_num * sizeof(ScalarType));

    int face_num = faces.size(0);
    int* d_face_num;
    cudaMalloc(&d_face_num, sizeof(int));
    cudaMemcpy(d_face_num, &face_num, sizeof(int), cudaMemcpyHostToDevice);

    int* d_img_size;
    cudaMalloc(&d_img_size, sizeof(int));
    cudaMemcpy(d_img_size, &img_size, sizeof(int), cudaMemcpyHostToDevice);

    compute_render<<<(face_num + threadnum - 1) / threadnum, threadnum>>>(
        d_compute_tri_area, d_faces, d_proj_xy, 
        d_total_gray_value, d_render_result, d_face_num, d_edge_len, d_img_size);
    compute_sobel<<<(img_size*img_size+ threadnum - 1)/threadnum,threadnum>>>( d_render_sobel_x, d_render_sobel_y, d_render_result, d_img_size);
    compute_sub<<<(img_size*img_size+ threadnum - 1)/threadnum,threadnum>>>( d_render_sobel_x, d_real_sobel_x, d_sobel_x_diff, d_img_size);
    compute_sub<<<(img_size*img_size+ threadnum - 1)/threadnum,threadnum>>>( d_render_sobel_y, d_real_sobel_y, d_sobel_y_diff, d_img_size);
    compute_sub<<<(img_size*img_size+ threadnum - 1)/threadnum,threadnum>>>( d_render_result, d_real_picture, d_render_diff, d_img_size);
    compute_sobel_weight<<<(img_size*img_size+ threadnum - 1)/threadnum,threadnum>>>( d_sobel_x_diff, d_sobel_y_diff, d_sobel_weight, d_img_size);
    compute_diff<<<(face_num + threadnum - 1) / threadnum, threadnum>>>( d_compute_tri_area, d_faces, d_proj_xy, d_total_gray_value, d_render_result, d_real_picture, 
        d_render_diff, d_points_diff, d_tri_area_diff, d_face_num, d_sobel_weight, d_points_sobel_diff, d_edge_len, d_img_size, d_points_diff_l1);

    torch::Tensor render_result = torch::from_blob(d_render_result, {img_size, img_size}, torch::kFloat).to(torch::kCUDA);
    render_result = render_result.clone();
    torch::Tensor render_diff = torch::from_blob(d_render_diff, {img_size, img_size}, torch::kFloat).to(torch::kCUDA);
    render_diff = render_diff.clone();
    torch::Tensor points_diff = torch::from_blob(d_points_diff, {points_num , 2}, torch::kFloat).to(torch::kCUDA);
    points_diff = points_diff.clone();
    torch::Tensor points_diff_l1 = torch::from_blob(d_points_diff_l1, {points_num , 2}, torch::kFloat).to(torch::kCUDA);
    points_diff = points_diff.clone();
    torch::Tensor tri_area_diff = torch::from_blob(d_tri_area_diff, {origin_faces_num}, torch::kFloat).to(torch::kCUDA);   
    tri_area_diff = tri_area_diff.clone();

    torch::Tensor points_sobel_diff = torch::from_blob(d_points_sobel_diff, {points_num , 2}, torch::kFloat).to(torch::kCUDA);
    points_sobel_diff = points_sobel_diff.clone();

    cudaFree(d_points_diff);
    cudaFree(d_points_diff_l1);
    cudaFree(d_tri_area_diff);
    cudaFree(d_face_num);
    cudaFree(d_render_result);
    cudaFree(d_render_diff);
    cudaFree(d_img_size);
    cudaFree(d_render_sobel_x);
    cudaFree(d_render_sobel_y);
    cudaFree(d_sobel_x_diff);
    cudaFree(d_sobel_y_diff);
    cudaFree(d_sobel_weight);
    cudaFree(d_points_sobel_diff);

    return {render_diff,points_diff,tri_area_diff,render_result,points_sobel_diff, points_diff_l1};
}

std::vector<torch::Tensor> real_sobel( torch::Tensor real_picture, int img_size)
{
    int threadnum = 256;
    ScalarType* d_real_picture = real_picture.data_ptr<ScalarType>();

    ScalarType* d_picture_sobel_x;
    cudaMalloc(&d_picture_sobel_x, img_size * img_size * sizeof(ScalarType));
    cudaMemset(d_picture_sobel_x, 0, img_size * img_size * sizeof(ScalarType));
    
    ScalarType* d_picture_sobel_y;
    cudaMalloc(&d_picture_sobel_y, img_size * img_size * sizeof(ScalarType));
    cudaMemset(d_picture_sobel_y, 0, img_size * img_size * sizeof(ScalarType));

    int* d_img_size;
    cudaMalloc(&d_img_size, sizeof(int));
    cudaMemcpy(d_img_size, &img_size, sizeof(int), cudaMemcpyHostToDevice);

    compute_sobel<<<(img_size*img_size+threadnum-1)/threadnum,threadnum>>>( d_picture_sobel_x, d_picture_sobel_y, d_real_picture, d_img_size);

    torch::Tensor sobel_x_result = torch::from_blob(d_picture_sobel_x, {img_size, img_size}, torch::kFloat).to(torch::kCUDA);
    sobel_x_result = sobel_x_result.clone();
    
    torch::Tensor sobel_y_result = torch::from_blob(d_picture_sobel_y, {img_size, img_size}, torch::kFloat).to(torch::kCUDA);
    sobel_y_result = sobel_y_result.clone();

    cudaFree(d_picture_sobel_x);
    cudaFree(d_picture_sobel_y);
    return {sobel_x_result,sobel_y_result};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("render", render);
    m.def("diff", diff);
    m.def("real_sobel",real_sobel);
}

      