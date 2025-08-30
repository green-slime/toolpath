#include <torch/extension.h>
#include "all_kernels.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("intersect", &intersect_object_parallel_v2, "求交，对扫掠体并行");
    m.def("intersect_with_height_grad", &intersect_with_height_grad, "计算几何loss");
    m.def("intersect_with_grad_for_render", &intersect_with_grad_for_render, "计算渲染loss");
}