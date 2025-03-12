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
using VectorXdual = Eigen::Matrix<dual, -1, 1, 0, -1, 1>;
// using autodiff::ArrayXreal2nd;
auto f(const ArrayXdual2nd& x) -> dual2nd {
    MatrixXdual2nd Q(2, 2);
    Q << 0.5, 0, 0, 1;
    ArrayXdual2nd b(2);
    b << 1, 0;
    return (0.5 * (x.matrix() - b.matrix()).transpose() * Q * (x.matrix() - b.matrix()))(0);
}


auto c(const ArrayXdual2nd& x) -> ArrayXdual2nd {
    ArrayXdual2nd res(1);
    res(0) = x(0) * x(0) + 2 * x(0) - x(1);
    return res;
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
// gauss newton
void newton_equality_minimize(const std::function<dual2nd(const ArrayXdual2nd&)>& f, decltype(c) c, ArrayXdual2nd& x, ArrayXdual2nd& lam,
                        double tol, const int& max_iter) {
    auto L = [f, c](const ArrayXdual2nd& x, const ArrayXdual2nd& lam) -> dual2nd {
        return f(x) + (lam.matrix().transpose() * c(x).matrix())(0);
    };
    Eigen::MatrixXd C(lam.size(), x.size());
    Eigen::MatrixXd H(x.size(), x.size());
    Eigen::MatrixXd A(x.size() + lam.size(), x.size() + lam.size());
    Eigen::VectorXd F(x.size() + lam.size());
    Eigen::VectorXd dx;

    try {
        C = jacobian(c, wrt(x), at(x));
        H = hessian(f, wrt(x), at(x));
        A.block(0, 0, x.size(), x.size()) = H;             // 左上块 H
        A.block(0, x.size(), x.size(), lam.size()) = C.transpose();  // 右上块 C^T
        A.block(x.size(), 0, lam.size(), x.size()) = C;             // 左下块 C
        A.block(x.size(), x.size(), lam.size(), lam.size()).setZero();       // 右下块 0 (自动填充为零)]]
        
        F << -gradient(L, wrt(x), at(x, lam)).matrix(), -c(x).matrix().cast<double>();
    } catch (std::exception& e) {
        std::cout << "newton minimize Exception: " << e.what() << std::endl;
        return;
    }
    int iter = 0;
    while (iter < max_iter) {
        // regularize_hessian(H);
        dx = A.inverse() * F;
        if (dx.norm() < tol) {
            break;
        }
        // double a = line_search(f, x, F.transpose()*dx, dx, 0.0001, 0.5);
        // x += a * dx.array().cast<dual2nd>();
        // x += dx.array().cast<dual2nd>();
        x += dx.block(0, 0, x.size(), 1).array().cast<dual2nd>();
        lam += dx.block(x.size(), 0, lam.size(), 1).array().cast<dual2nd>();

        C = jacobian(c, wrt(x), at(x));
        H = hessian(f, wrt(x), at(x));
        // regularize_hessian(H);
        A.block(0, 0, x.size(), x.size()) = H;             // 左上块 H
        A.block(0, x.size(), x.size(), lam.size()) = C.transpose();  // 右上块 C^T
        A.block(x.size(), 0, lam.size(), x.size()) = C;             // 左下块 C
        A.block(x.size(), x.size(), lam.size(), lam.size()).setZero();       // 右下块 0 (自动填充为零)]]
        
        F << -gradient(L, wrt(x), at(x, lam)).matrix(), -c(x).matrix().cast<double>();

        iter++;
    }
    std::cout << "newton minimize: " << iter << std::endl << "napla f(x):\n" << F << std::endl;
}

int main() {
    // ArrayXdual2nd x(2);
    // x << 0, 0;

    ArrayXdual2nd x(2);
    x << -3, 2;
    ArrayXdual2nd lam(1);
    lam << 0;
    
    newton_equality_minimize(f, c, x, lam, 1e-6, 100);
    std::cout << "x:\n" << x.matrix().cast<double>() << std::endl;
}