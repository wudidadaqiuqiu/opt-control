// C++ includes
#include <iostream>

// autodiff include
#include <autodiff/forward/real.hpp>
#include <autodiff/forward/real/eigen.hpp>
#include <autodiff/forward/dual.hpp>
#include <Eigen/Eigenvalues>
using namespace autodiff;

// The vector function for which the Jacobian is needed
VectorXreal f(const VectorXreal& x)
{
    return x * x.sum();
}

dual f1(dual x)
{
    return 1 + x + x*x + 1/x + log(x);
}

VectorXreal pendulum_dynamics(const VectorXreal& x) {
    const double g = 9.81;
    const double l = 1.0;
    VectorXreal F(2, 1);
    F(0) = x(1);
    F(1) = -g/l * sin(x(0));
    return F;
}


VectorXreal pendulum_dynamics_rk4(const VectorXreal& xk, const real& h) {
    VectorXreal f1 = pendulum_dynamics(xk);
    VectorXreal f2 = pendulum_dynamics(xk + 0.5*h*f1);
    VectorXreal f3 = pendulum_dynamics(xk + 0.5*h*f2);
    VectorXreal f4 = pendulum_dynamics(xk + h*f3);
    return xk + (h/6.0)*(f1 + 2*f2 + 2*f3 + f4);
}

VectorXreal test_jacobian_dynamic(const VectorXreal& x) {
    const double g = 9.81;
    const double l = 1.0;
    VectorXreal F(3);
    F(0) = x(1);
    F(1) = -g/l * sin(x(0));
    F(2) = x(1) + 1.0;
    return F;
}

auto is_mat_pd(const Eigen::MatrixXd& mat) -> bool {
    // 使用 Cholesky 分解检查矩阵是否正定
    Eigen::LLT<Eigen::MatrixXd> llt(mat);
    
    // 判断分解是否成功（Llt.success() 会返回 true 如果成功）
    return llt.info() == Eigen::Success;
}

auto regularize_mat(Eigen::MatrixXd& mat, const double& step=1.0) -> void {
    while (!is_mat_pd(mat)) {
        mat += step * Eigen::MatrixXd::Identity(mat.rows(), mat.cols());
    }
}

auto line_search(const std::function<VectorXreal(const VectorXreal&)>& f, const VectorXreal& x, 
                const VectorXreal& df, const VectorXreal& dx, double beta, double c) -> double {
    double a = 1.0;
    while (f(x + a * dx).norm() > (f(x) + beta * a * df).norm()){
        a *= c;
    }
    return a;
}

void newton_find_root(const std::function<VectorXreal(const VectorXreal&)>& f, VectorXreal& x, 
                        double tol, const int& max_iter) {
    VectorXreal F;
    Eigen::MatrixXd J;
    VectorXreal dx;
    try {
        F = f(x);
        J = jacobian(f, wrt(x), at(x));
    } catch (std::exception& e) {
        std::cout << "newtow find root Exception: " << e.what() << std::endl;
        return;
    }
    int iter = 0;
    while (F.norm() > tol && iter < max_iter) {
        // regularize_mat(J);
        dx = - J.inverse() * F;
        // double a = line_search(f, x, J*dx, dx, 0.0001, 0.5);
        // x += a * dx;
        x += dx;
        F = f(x);
        J = jacobian(f, wrt(x), at(x));
        iter++;
    }
    std::cout << "newton find root: " << iter << std::endl << "f(x):\n" << F << std::endl;
}


int main()
{
    using Eigen::MatrixXd;

    VectorXreal x(2);                           // the input vector x with 5 variables
    x << 1, 2;                         // x = [1, 2, 3, 4, 5]

    VectorXreal F;                              // the output vector F = f(x) evaluated together with Jacobian matrix below
    MatrixXd TestJ = jacobian(test_jacobian_dynamic, wrt(x), at(x));
    // Jacobian matrix:
    //        0        1
    // -5.30037       -0
    //        0        1
    std::cout << "Jacobian matrix:\n" << TestJ << std::endl;

    // assert error
    // newton_find_root(test_jacobian_dynamic, x, 1e-6, 10);
    newton_find_root(pendulum_dynamics, x, 1e-6, 100);
    std::cout << "pendulum_dynamics fixed point = \n" << x << std::endl;    // print the solution x
    
    VectorXreal xk(2);
    xk << 0.0, 0.0;
    MatrixXd J = jacobian(pendulum_dynamics_rk4, wrt(xk), at(xk, 0.01), F); // evaluate the output vector F and the Jacobian matrix dF/dx


    std::cout << "F = \n" << F << std::endl;    // print the evaluated output vector F
    std::cout << "J = \n" << J << std::endl;    // print the evaluated Jacobian matrix dF/dx

    Eigen::EigenSolver<Eigen::MatrixXd> solver(J);
    
    std::cout << "Eigenvalues: \n" << solver.eigenvalues() << std::endl;
}