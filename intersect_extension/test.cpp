#include<ceres/jet.h>
#include <cassert>

int main() {
    ceres::Jet<double, 3> x(2.0, 0); // x = 2, dx/dx = 1
    ceres::Jet<double, 3> y(3.0, 1); // y = 3, dy/dy = 1
    ceres::Jet<double, 3> z = x * y + ceres::sin(x);
    std::cout << "Value: " << z.a << ", Grad: [" << z.v << std::endl;
    return 0;
}