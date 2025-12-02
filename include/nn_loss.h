//
// Created by rudri on 10/11/2020.
//

#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_LOSS_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_LOSS_H
#include "nn_interfaces.h"
#include <algorithm>
namespace utec::neural_network {

template<typename T>
class MSELoss final: public ILoss<T, 2> {
    private:
    const Tensor<T,2>& y_pred_;
    const Tensor<T,2>& y_true_;
    int n; 

    public:
    MSELoss(const Tensor<T,2>& y_prediction, const Tensor<T,2>& y_true)
        : y_pred_(y_prediction), y_true_(y_true) {
        auto shape = y_pred_.shape();
        n = shape[0] * shape[1];
    }

    T loss() const override {
        T sum_sq_diff = 0;
        for (auto p_it = y_pred_.cbegin(), t_it = y_true_.cbegin(); 
                p_it != y_pred_.cend(); ++p_it, ++t_it) 
        {
            T diff = *p_it - *t_it;
            sum_sq_diff += diff * diff;
        }
        return sum_sq_diff / static_cast<T>(n);
    }

    Tensor<T,2> loss_gradient() const override {
        return (y_pred_ - y_true_) * (T(2) / static_cast<T>(n));
    }
};

template<typename T>
class BCELoss final: public ILoss<T, 2> {
    private:
    const Tensor<T,2>& y_pred_;
    const Tensor<T,2>& y_true_;
    size_t n;
    const T epsilon = 1e-9;

  public:
    BCELoss(const Tensor<T,2>& y_prediction, const Tensor<T,2>& y_true)
      : y_pred_(y_prediction), y_true_(y_true) {
        auto shape = y_pred_.shape();
        n = shape[0] * shape[1];
    }

    T loss() const override {
        T sum_loss = 0;
        for (auto p_it = y_pred_.cbegin(), t_it = y_true_.cbegin(); 
             p_it != y_pred_.cend(); ++p_it, ++t_it) 
        {
            T y = *t_it;
            T p = std::clamp(*p_it, epsilon, T(1) - epsilon);
            sum_loss += -(y * std::log(p) + (T(1) - y) * std::log(T(1) - p));
        }
        return sum_loss / static_cast<T>(n);
    }

    Tensor<T,2> loss_gradient() const override {
        Tensor<T,2> grad(y_pred_.shape());
        T N_T = static_cast<T>(n);

        auto g_it = grad.begin();
        for (auto p_it = y_pred_.cbegin(), t_it = y_true_.cbegin(); 
             p_it != y_pred_.cend(); ++p_it, ++t_it, ++g_it) 
        {
            T y = *t_it;
            T p = std::clamp(*p_it, epsilon, T(1) - epsilon);
            *g_it = (T(1) / N_T) * (p - y) / (p * (T(1) - p));
        }
        return grad;
    }
};
}

#endif //PROG3_NN_FINAL_PROJECT_V2025_01_LOSS_H
