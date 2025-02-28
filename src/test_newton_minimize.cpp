// C++ includes
#include <iostream>

// autodiff include
#include <autodiff/forward/real.hpp>
#include <autodiff/forward/real/eigen.hpp>
#include <autodiff/forward/dual.hpp>
#include <Eigen/Eigenvalues>
using namespace autodiff;

using ArrayXdual2nd = Eigen::Array<dual2nd, -1, 1>;
using MatrixXdual2nd = Eigen::Matrix<dual2nd, -1, -1, 0, -1, -1>;
// using autodiff::ArrayXreal2nd;
auto f(const ArrayXdual2nd& x) -> dual2nd {
    MatrixXdual2nd Q(2, 2);
    Q << 0.5, 0, 0, 1;
    ArrayXdual2nd b(2);
    b << 1, 0;
    return (0.5 * (x.matrix() - b.matrix()).transpose() * Q * (x.matrix() - b.matrix()))(0);
}


auto f2(const ArrayXdual2nd& x) -> dual2nd {
    using detail::pow;
    return pow(x(0), 4) + pow(x(0), 3) - pow(x(0), 2) - x(0);
}

auto is_mat_pd(const Eigen::MatrixXd& mat) -> bool {
    // 使用 Cholesky 分解检查矩阵是否正定
    Eigen::LLT<Eigen::MatrixXd> llt(mat);
    
    // 判断分解是否成功（Llt.success() 会返回 true 如果成功）
    return llt.info() == Eigen::Success;
}

auto regularize_hessian(Eigen::MatrixXd& mat, const double& step=1.0) -> void {
    while (!is_mat_pd(mat)) {
        mat += step * Eigen::MatrixXd::Identity(mat.rows(), mat.cols());
    }
}

auto line_search(const std::function<dual2nd(const ArrayXdual2nd&)>& f, const ArrayXdual2nd& x, 
                const double& df, const Eigen::VectorXd& dx, double beta, double c) -> double {
    double a = 1.0;
    while (f(x + a * dx.array().cast<dual2nd>()) > f(x) + beta * a * df){
        a *= c;
    }
    return a;
}

//  error: static assertion failed: Real<N, T> is optimized for higher-order **directional** derivatives. You're possibly trying to use it for computing higher-order **cross** derivatives (e.g., `derivative(f, wrt(x, x, y), at(x, y))`) which is not supported by Real<N, T>. Use Dual<T, G> instead (e.g., `using dual4th = HigherOrderDual<4>;`)
//   966 |     static_assert(order == 1,
//       |                   ~~~~~~^~~~
// 高阶导需要使用Array，dual 用于求向量，real用于求方向导数
void newton_minimize(const std::function<dual2nd(const ArrayXdual2nd&)>& f, ArrayXdual2nd& x, 
                        double tol, const int& max_iter) {
    Eigen::VectorXd F;
    Eigen::MatrixXd H;
    
    Eigen::VectorXd dx;
    try {
        F = gradient(f, wrt(x), at(x));
        H = hessian(f, wrt(x), at(x));
    } catch (std::exception& e) {
        std::cout << "newtow minimize Exception: " << e.what() << std::endl;
        return;
    }
    int iter = 0;
    while (F.norm() > tol && iter < max_iter) {
        regularize_hessian(H);
        dx = - H.inverse() * F;
        double a = line_search(f, x, F.transpose()*dx, dx, 0.0001, 0.5);
        x += a * dx.array().cast<dual2nd>();
        // x += dx.array().cast<dual2nd>();
        F = gradient(f, wrt(x), at(x));
        H = hessian(f, wrt(x), at(x));
        iter++;
    }
    std::cout << "newton minimize: " << iter << std::endl << "napla f(x):\n" << F << std::endl;
}

int main() {
    // ArrayXdual2nd x(2);
    // x << 0, 0;
    ArrayXdual2nd x(1);
    x << 0;
    newton_minimize(f2, x, 1e-6, 10);
    std::cout << "x:\n" << x.matrix().cast<double>() << std::endl;
}