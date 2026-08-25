// test_psf_zero_ceres.cpp

// Verifies psf_zero_ceres.hpp against finite differences and a real

// ceres::Solve(). Build (needs libceres-dev, libeigen3-dev):

//   g++ -std=c++17 -O2 test_psf_zero_ceres.cpp -I/usr/include/eigen3 \

//       -lceres -lglog -o test_psf_zero_ceres && ./test_psf_zero_ceres

#include "psf_zero_ceres.hpp"

#include <ceres/ceres.h>

#include <vector>

#include <random>

#include <cstdio>

#include <cmath>

using namespace psf;

// ---- A simple linear inner cost, used to check ZeroClampCostWrapper's Jacobian ----

class LinearCost final : public ceres::CostFunction {

public:

    LinearCost(const Eigen::Matrix<double,3,2>& A, const Eigen::Vector3d& b) : A_(A), b_(b) {

        set_num_residuals(3);

        mutable_parameter_block_sizes()->push_back(2);

    }

    bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const override {

        Eigen::Map<const Eigen::Vector2d> p(parameters[0]);

        Eigen::Map<Eigen::Vector3d> r(residuals);

        r = A_ * p + b_;

        if (jacobians && jacobians[0]) {

            Eigen::Map<Eigen::Matrix<double,3,2,Eigen::RowMajor>> J(jacobians[0]);

            J = A_;

        }

        return true;

    }

private:

    Eigen::Matrix<double,3,2> A_;

    Eigen::Vector3d b_;

};

struct LineResidual {

    LineResidual(double x, double y) : x_(x), y_(y) {}

    template <typename T>

    bool operator()(const T* const params, T* residual) const {

        residual[0] = params[0] * T(x_) + params[1] - T(y_);

        return true;

    }

    double x_, y_;

};

int main() {

    // ---- 1. ZeroClampLoss: rho[1]/rho[2] vs finite differences ----

    printf("=== ZeroClampLoss: rho[1], rho[2] vs finite differences ===\n");

    double tau = 1.7;

    ZeroClampLoss loss(tau);

    double max_rel_err1 = 0.0, max_rel_err2 = 0.0;

    for (double s : {0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0}) {

        double rho[3], rho_p[3], rho_m[3];

        double h = 1e-6 * std::max(1.0, s);

        loss.Evaluate(s, rho);

        loss.Evaluate(s + h, rho_p);

        loss.Evaluate(s - h, rho_m);

        double fd1 = (rho_p[0] - rho_m[0]) / (2 * h);

        double fd2 = (rho_p[1] - rho_m[1]) / (2 * h);

        max_rel_err1 = std::max(max_rel_err1, std::abs(rho[1]-fd1)/std::abs(fd1));

        max_rel_err2 = std::max(max_rel_err2, std::abs(rho[2]-fd2)/std::abs(fd2));

        printf("  s=%6.2f  rho1=%.8f (fd=%.8f)  rho2=%.8f (fd=%.8f)\n",

               s, rho[1], fd1, rho[2], fd2);

    }

    printf("max relative error: rho[1]=%.2e  rho[2]=%.2e\n\n", max_rel_err1, max_rel_err2);

    // ---- 2. ZeroClampCostWrapper: Jacobian vs finite differences ----

    printf("=== ZeroClampCostWrapper: Jacobian vs finite differences ===\n");

    Eigen::Matrix<double,3,2> A;

    A << 0.5, -1.2,  1.3, 0.4,  -0.7, 0.9;

    Eigen::Vector3d b(3.0, -2.0, 2.5);

    double p0[2] = {0.3, -0.4};

    double tau2 = 1.5;

    ZeroClampCostWrapper wrapped(new LinearCost(A, b), tau2, true);

    double residuals0[3], jac_storage[6];

    double* params[1] = {p0};

    double* jacobians[1] = {jac_storage};

    wrapped.Evaluate(params, residuals0, jacobians);

    double eps = 1e-6, J_fd[6];

    for (int j = 0; j < 2; ++j) {

        double pp[2] = {p0[0], p0[1]}, pm[2] = {p0[0], p0[1]};

        pp[j] += eps; pm[j] -= eps;

        double rp[3], rm[3];

        double* pparams_p[1] = {pp};

        double* pparams_m[1] = {pm};

        ZeroClampCostWrapper w1(new LinearCost(A, b), tau2, true);

        ZeroClampCostWrapper w2(new LinearCost(A, b), tau2, true);

        w1.Evaluate(pparams_p, rp, nullptr);

        w2.Evaluate(pparams_m, rm, nullptr);

        for (int i = 0; i < 3; ++i) J_fd[i*2+j] = (rp[i] - rm[i]) / (2*eps);

    }

    double max_err = 0.0;

    for (int i = 0; i < 6; ++i) max_err = std::max(max_err, std::abs(jac_storage[i]-J_fd[i]));

    printf("max |analytic - finite_diff| = %.3e\n\n", max_err);

    // ---- 3. End-to-end ceres::Solve() with outliers ----

    printf("=== End-to-end line fit with 4 large outliers (60 points) ===\n");

    std::mt19937 rng(7);

    std::normal_distribution<double> noise(0.0, 0.15);

    std::vector<double> xs, ys;

    for (int i = 0; i < 60; ++i) {

        double x = i * 0.2 - 6.0;

        xs.push_back(x);

        ys.push_back(2.0*x + 1.0 + noise(rng));

    }

    for (int idx : {5, 20, 35, 50}) ys[idx] += (idx % 2 == 0 ? 1 : -1) * 15.0;

    auto fit = [&](ceres::LossFunction* lf, const char* label) {

        double params[2] = {0.0, 0.0};

        ceres::Problem problem;

        for (size_t i = 0; i < xs.size(); ++i) {

            auto* cost = new ceres::AutoDiffCostFunction<LineResidual, 1, 2>(new LineResidual(xs[i], ys[i]));

            problem.AddResidualBlock(cost, lf, params);

        }

        ceres::Solver::Options options;

        options.linear_solver_type = ceres::DENSE_QR;

        ceres::Solver::Summary summary;

        ceres::Solve(options, &problem, &summary);

        double err = std::sqrt(std::pow(params[0]-2.0, 2) + std::pow(params[1]-1.0, 2));

        printf("  %-20s slope=%.4f intercept=%.4f  ||params-true||=%.4f\n", label, params[0], params[1], err);

    };

    printf("  (true: slope=2.0000 intercept=1.0000)\n");

    fit(nullptr, "Plain L2");

    fit(new ZeroClampLoss(1.0), "ZeroClampLoss (fixed)");

    return 0;

}
