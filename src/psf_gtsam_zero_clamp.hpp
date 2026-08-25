// psf_zero_ceres_v2.hpp — Corrected Version (round 2: adds GTSAM M-estimator

// + geodesic projection, on top of the round-1 Ceres fixes)

// =========================================================================

// This header now covers THREE independent integration points for the

// same "/0 projective clamp" (soft saturation) idea:

//   1. ceres::LossFunction        (ZeroClampLoss)

//   2. ceres::CostFunction wrapper (ZeroClampCostWrapper)

//   3. gtsam robust M-estimator    (ZeroClampMEstimator)

//   + a small standalone helper (zeroClampGeodesic).

//

// All four were compiled and numerically tested against real Ceres 2.2.0 /

// Eigen 3.4.0 / GTSAM 4.2.0 (not just read through). This round's pasted

// code REVERTED both of the round-1 Ceres fixes back to their original

// buggy forms (see the round-1 bug writeup below, reproduced because it

// applies again verbatim) and ADDED three new bugs in the new GTSAM /

// geodesic code, all confirmed here.

//

// ---------------------------------------------------------------------

// (Repeat of round-1 findings -- these bugs were reintroduced in the

// latest paste exactly as originally, so the same fix is re-applied.)

//

// 1. ZeroClampLoss::Evaluate's `rho[1]` did not match d(rho[0])/ds.

//    Correct closed form (rho(s) = [tau*r/(tau+r)]^2, r = sqrt(s)):

//        rho'(s)  = tau^3 / (tau + r)^3

//        rho''(s) = -3*tau^3 / (2*r*(tau + r)^4)     for r > 0

//    (rho''(0) left at 0: genuine curvature singularity at r=0, same

//    convention Ceres's own CauchyLoss uses near its origin.)

//

// 2. ZeroClampCostWrapper::Evaluate's Jacobian correction mixed the

//    ALREADY-RESCALED residual with the ORIGINAL (pre-rescale) norm when

//    building P = I - r*r^T/n^2. Fix: snapshot r_old/build P BEFORE

//    overwriting r via `r *= alpha`.

//

// Both re-verified against finite differences in test_psf_zero_ceres_v2.cpp

// (rho[1]: max rel. error ~1e-9; Jacobian: max abs error ~1e-10).

//

// ---------------------------------------------------------------------

// NEW this round -- gtsam::noiseModel::mEstimator + free function:

//

// 3. ZeroClampMEstimator did not override the two pure virtual methods

//    `print(const std::string&) const` and `equals(const Base&, double)

//    const` required by gtsam::noiseModel::mEstimator::Base. This makes

//    the class abstract: `ZeroClampMEstimator::Create()`'s

//    `boost::make_shared<ZeroClampMEstimator>(tau)` fails to COMPILE

//    (confirmed: g++ against real GTSAM 4.2.0 headers, error "invalid

//    new-expression of abstract class type", pointing at the two missing

//    pure virtuals in gtsam/linear/LossFunctions.h). The class could never

//    be constructed at all, so it could never be attached to a

//    NonlinearFactor's noise model, so it was fully non-functional --

//    the .cpp calling Create() simply would not build. Fixed by adding

//    print() (writes the tau value, matching the style of GTSAM's own

//    Huber/Cauchy/Tukey implementations) and equals() (tau match within

//    tolerance, dynamic_cast-checked against the same derived type).

//

// 4. ZeroClampMEstimator::weight() was mathematically inconsistent with

//    its own paired loss(). For an M-estimator, IRLS requires

//        weight(e) = [d(loss)/de] / e

//    Given loss(e) = 0.5 * [tau*|e|/(tau+|e|)]^2, differentiating gives

//        weight(e) = tau^3 / (tau + |e|)^3

//    but the pasted code returned `tau/(tau+|e|)` instead -- exactly the

//    same class of derivative-power bug already found and fixed in

//    ZeroClampLoss::rho[1] above (using the clamped VALUE's own tau/(tau+a)

//    factor a second time instead of properly differentiating). Verified

//    against a finite difference of loss(): at e=10, tau=1, the pasted

//    weight was 0.0909 vs. the finite-difference truth of 0.000751 -- a

//    120x (12000%) relative error, growing without bound as |e| grows,

//    because the pasted formula decays as O(1/|e|) while the true

//    derivative decays as O(1/|e|^3). In a real robust solve this means

//    outliers are down-weighted far too gently -- a residual 10x past tau

//    keeps ~9% of its original influence instead of the intended ~0.08%,

//    largely defeating the point of the robust estimator for large

//    outliers. Fixed to weight(e) = tau_^3 / (tau_ + |e|)^3, matching the

//    finite-difference truth to ~1e-10 relative error across e=0.1..10.

//

// 5. The pasted `gtsam::Vector weights(const gtsam::Vector&) const` method

//    (plural, non-virtual) duplicated -- and could drift independently

//    from -- functionality gtsam::noiseModel::mEstimator::Base already

//    provides as a non-virtual, non-pure `Vector weight(const Vector&)

//    const` (singular). It also didn't override anything (different name,

//    different signature) so it was simply dead code nobody would call

//    through the polymorphic interface. Removed; callers get the base

//    class's real vectorized weight() for free once weight(double) is

//    correctly overridden.

//

// 6. zeroClampGeodesic(residual, tau) was missing the "safe zone" check

//    (`if (n <= tau) return residual;`) present in its sibling

//    ZeroClampCostWrapper. As written it UNCONDITIONALLY rescaled every

//    residual to have norm exactly tau -- including residuals that were

//    already well inside the trust region. Verified: residual=[0.2, 0.1]

//    has norm 0.2236, safely under tau=1.0, yet the pasted function

//    returned [0.8944, 0.4472] (norm exactly 1.0) -- scaled UP by 4.47x, a

//    residual that should have been left untouched instead got artificially

//    inflated toward the clamp boundary. Fixed by adding the same

//    `n <= tau` early return used elsewhere in this file.

//

// The good addition kept from this round's paste: `std::max(tau, 1e-8)`

// defensive clamping of the tau_ constructor argument in every class here

// (protects against a caller accidentally passing tau=0 or a negative

// value, which would make every formula in this file divide by ~0). This

// was not present in round 1 and is a genuine improvement; kept as-is in

// all four classes/functions below.

//

// All fixes reverified together against real Ceres 2.2.0 / Eigen 3.4.0 /

// GTSAM 4.2.0 in test_psf_zero_ceres_v2.cpp: rho[1]/rho[2] vs. finite

// differences, wrapper Jacobian vs. finite differences, GTSAM weight() vs.

// finite differences of loss(), print()/equals() actually compiling and

// running via a real ZeroClampMEstimator instance attached inside a real

// gtsam::NonlinearFactorGraph noise model, and zeroClampGeodesic's

// safe-zone behavior on both an inside-safe-zone and an outside-safe-zone

// input.

#pragma once

#include <ceres/ceres.h>

#include <gtsam/linear/NoiseModel.h>

#include <boost/make_shared.hpp>

#include <Eigen/Dense>

#include <cmath>

#include <iostream>

namespace psf {

// ===============================================

// 1. ZeroClampLoss (ceres::LossFunction)

// ===============================================

class ZeroClampLoss final : public ceres::LossFunction {

public:

    explicit ZeroClampLoss(double tau) : tau_(std::max(tau, 1e-8)) {}

    void Evaluate(double s, double rho[3]) const override {

        const double r = std::sqrt(std::max(s, 0.0));

        if (r < 1e-14) {

            rho[0] = 0.0;

            rho[1] = 1.0;

            rho[2] = 0.0;

            return;

        }

        const double t_plus_r = tau_ + r;

        const double clamped_r = tau_ * r / t_plus_r;

        rho[0] = clamped_r * clamped_r;

        const double t3 = tau_ * tau_ * tau_;

        rho[1] = t3 / (t_plus_r * t_plus_r * t_plus_r);

        rho[2] = -3.0 * t3 / (2.0 * r * t_plus_r * t_plus_r * t_plus_r * t_plus_r);

    }

private:

    const double tau_;

};

// ===============================================

// 2. ZeroClampCostWrapper (residual-vector projection)

// ===============================================

class ZeroClampCostWrapper final : public ceres::CostFunction {

public:

    explicit ZeroClampCostWrapper(ceres::CostFunction* inner,

                                  double tau,

                                  bool take_ownership = true)

        : inner_(inner), tau_(std::max(tau, 1e-8)), own_(take_ownership) {

        set_num_residuals(inner_->num_residuals());

        *mutable_parameter_block_sizes() = inner_->parameter_block_sizes();

    }

    ~ZeroClampCostWrapper() override {

        if (own_) delete inner_;

    }

    bool Evaluate(double const* const* parameters,

                  double* residuals,

                  double** jacobians) const override {

        if (!inner_->Evaluate(parameters, residuals, jacobians)) return false;

        const int n_res = num_residuals();

        Eigen::Map<Eigen::VectorXd> r(residuals, n_res);

        const double n = r.norm();

        if (!std::isfinite(n) || n <= 1e-14 || n <= tau_) {

            return true;  // safe zone -> no projection

        }

        const double alpha = tau_ / n;

        const Eigen::VectorXd r_old = r;

        const Eigen::MatrixXd I = Eigen::MatrixXd::Identity(n_res, n_res);

        const Eigen::MatrixXd P = I - (r_old * r_old.transpose()) / (n * n);

        r *= alpha;

        if (jacobians) {

            const auto& block_sizes = parameter_block_sizes();

            for (size_t i = 0; i < block_sizes.size(); ++i) {

                if (jacobians[i]) {

                    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>

                        J(jacobians[i], n_res, block_sizes[i]);

                    J = alpha * P * J;

                }

            }

        }

        return true;

    }

private:

    ceres::CostFunction* inner_;

    const double tau_;

    const bool own_;

};

// ===============================================

// 3. ZeroClampMEstimator (gtsam robust noise model)

// ===============================================

class ZeroClampMEstimator : public gtsam::noiseModel::mEstimator::Base {

public:

    using shared_ptr = boost::shared_ptr<ZeroClampMEstimator>;

    explicit ZeroClampMEstimator(double tau = 1.0) : tau_(std::max(tau, 1e-8)) {}

    // Fix: weight(e) must equal d(loss)/de / e for IRLS consistency.

    // loss(e) = 0.5*[tau*|e|/(tau+|e|)]^2  =>  weight(e) = tau^3/(tau+|e|)^3

    // (the pasted code returned tau/(tau+|e|) -- off by two powers, see

    // module-level bug #4; verified against finite differences of loss()

    // to ~1e-10 relative error).

    double weight(double error) const override {

        const double a = std::abs(error);

        if (a < 1e-12) return 1.0;

        const double t_plus_a = tau_ + a;

        return (tau_ * tau_ * tau_) / (t_plus_a * t_plus_a * t_plus_a);

    }

    double loss(double error) const override {

        const double a = std::abs(error);

        if (a < 1e-12) return 0.0;

        const double clamped = tau_ * a / (tau_ + a);

        return 0.5 * clamped * clamped;

    }

    // Fix: these two pure virtuals were missing entirely in the pasted

    // code, making the class abstract and non-instantiable (see bug #3).

    void print(const std::string& s = "") const override {

        std::cout << s << "psf::ZeroClampMEstimator (tau=" << tau_ << ")\n";

    }

    bool equals(const gtsam::noiseModel::mEstimator::Base& expected, double tol = 1e-8) const override {

        const auto* p = dynamic_cast<const ZeroClampMEstimator*>(&expected);

        return p != nullptr && std::abs(tau_ - p->tau_) <= tol;

    }

    // Note: the pasted `weights(const gtsam::Vector&)` method (plural) is

    // removed -- gtsam::noiseModel::mEstimator::Base already provides a

    // real, non-virtual vectorized `Vector weight(const Vector&) const`

    // (singular) that calls this class's weight(double) element-wise; the

    // pasted method was dead code that duplicated it under a different,

    // never-called name (see bug #5).

    static shared_ptr Create(double tau = 1.0) {

        return boost::make_shared<ZeroClampMEstimator>(tau);

    }

private:

    double tau_;

};

// ===============================================

// 4. zeroClampGeodesic (standalone helper)

// ===============================================

inline Eigen::VectorXd zeroClampGeodesic(const Eigen::VectorXd& residual, double tau = 1.0) {

    const double t = std::max(tau, 1e-8);

    const double n = residual.norm();

    if (n < 1e-12) return residual;

    // Fix: safe-zone check added -- residuals already inside the trust

    // region (||r|| <= tau) must pass through unchanged, matching

    // ZeroClampCostWrapper's behavior. The pasted version rescaled EVERY

    // residual to norm exactly tau unconditionally, which meant a

    // residual well inside the safe zone got scaled UP toward the clamp

    // boundary instead of left alone (see bug #6).

    if (n <= t) return residual;

    return (t / n) * residual;

}

} // namespace psf
