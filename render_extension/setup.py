"""
使用 python setup.py install 进行相应 .cu 文件的注册
"""
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
setup(name='render_extension',
      ext_modules=[
            CUDAExtension(
                  name='render_extension', 
                  sources=['new_render.cu'],
                  include_dirs=[], 
                  extra_compile_args={'cxx': ['-O3', '-std=c++14', '-march=native', '-mtune=native'],
                                      'nvcc': ['-O3', '-std=c++14', '-U__CUDA_NO_HALF_OPERATORS__', '-U__CUDA_NO_HALF_CONVERSIONS__', '-U__CUDA_NO_HALF2_OPERATORS__']})
      ],
      cmdclass={'build_ext': BuildExtension})