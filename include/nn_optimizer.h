//
// Created by rudri on 10/11/2020.
//

#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_OPTIMIZER_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_OPTIMIZER_H
#include "nn_interfaces.h"
namespace utec::neural_network {
template<typename T>
class SGD final : public IOptimizer<T> {
    T learning_rate_;

    public:
    explicit SGD(T learning_rate = 0.01) : learning_rate_(learning_rate) {}

    void update(Tensor<T, 2>& params, const Tensor<T, 2>& grads) override {
        // params = params - lr * grads
        params = params - (grads * learning_rate_);
    }
};
template<typename T>
class Adam final : public IOptimizer<T> {
    T lr_, beta1_, beta2_, epsilon_;
    size_t t_ = 0; 

    struct State {
        Tensor<T, 2> m;
        Tensor<T, 2> v;
        State(const std::array<int, 2>& shape) : m(shape), v(shape) {
            m.fill(T(0));
            v.fill(T(0));
        }
    };

    std::map<T*, State> cache_;

    public:
    explicit Adam(T learning_rate = 0.001, T beta1 = 0.9, T beta2 = 0.999, T epsilon = 1e-8)
        : lr_(learning_rate), beta1_(beta1), beta2_(beta2), epsilon_(epsilon) {}

    void step() override {
        t_++; 
    }

    void update(Tensor<T, 2>& params, const Tensor<T, 2>& grads) override {
        if (t_ == 0) t_ = 1; 

        T* key = &params(0, 0); 
        
        if (cache_.find(key) == cache_.end()) {
            cache_.emplace(key, State(params.shape()));
        }
        State& s = cache_.at(key);

        auto m_it = s.m.begin();
        auto v_it = s.v.begin();
        auto p_it = params.begin();
        auto g_it = grads.cbegin();

        for (; p_it != params.end(); ++p_it, ++g_it, ++m_it, ++v_it) {
            T g = *g_it;

            *m_it = beta1_ * *m_it + (T(1) - beta1_) * g;
            
            *v_it = beta2_ * *v_it + (T(1) - beta2_) * g * g;

            // bias
            T m_hat = *m_it / (T(1) - std::pow(beta1_, t_));
            T v_hat = *v_it / (T(1) - std::pow(beta2_, t_));

            *p_it -= lr_ * m_hat / (std::sqrt(v_hat) + epsilon_);
        }
    }
};
}
#endif //PROG3_NN_FINAL_PROJECT_V2025_01_OPTIMIZER_H
