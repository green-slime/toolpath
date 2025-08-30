"""
使用 python setup.py install 进行相应 .cu 文件的注册
"""
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(name='new_intersect_extension',
      ext_modules=[
            CUDAExtension(
                  name='new_intersect_extension', 
                  sources=[
                      'new_compute.cu',        # pybind11注册文件
                      'new_intersect.cu',      # 新的intersect实现
                      'new_height_grad.cu',    # 高度梯度计算
                      'new_render_grad.cu'     # 渲染梯度计算
                  ],
                  include_dirs=[], 
                  extra_compile_args={
                      'cxx': ['-O3', '-std=c++14', '-march=native', '-mtune=native'],
                      'nvcc': ['-O3', '-std=c++14', 
                              '-U__CUDA_NO_HALF_OPERATORS__', 
                              '-U__CUDA_NO_HALF_CONVERSIONS__', 
                              '-U__CUDA_NO_HALF2_OPERATORS__']
                  })
      ],
      cmdclass={'build_ext': BuildExtension})