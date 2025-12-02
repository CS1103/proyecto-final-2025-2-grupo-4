//
// Created by rudri on 10/11/2020.
//

#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_ACTIVATION_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_ACTIVATION_H
#include <algorithm>
#include <cmath>
#include "nn_interfaces.h"
using namespace std;
namespace utec::neural_network {

template<typename T>
class ReLU final : public ILayer<T> {
    private:
    Tensor<T,2> temp_{{0,0}}; 

    public:
    Tensor<T,2> forward(const Tensor<T,2>& z) override {
        temp_ = z; 
        Tensor<T,2> result = z; 

        for (auto& val : result) {
            val = std::max(T(0), val);
        }
        return result;
    }

    Tensor<T,2> backward(const Tensor<T,2>& g) override {
        Tensor<T,2> dZ = g; 
        
        auto it = temp_.cbegin();
        for (auto dz_it = dZ.begin(); dz_it != dZ.end(); ++dz_it, ++it) {
            if (*it <= T(0)) {
                *dz_it = T(0); 
            }
        }
        return dZ;
    }
};

template<typename T>
class Sigmoid final : public ILayer<T> {
    private:
    Tensor<T,2> temp{{0,0}}; 

    public:
    Tensor<T,2> forward(const Tensor<T,2>& z) override {
        Tensor<T,2> result = z; 

        for (auto& val : result) {
            val = T(1) / (T(1) + std::exp(-val));
        }
        temp = result; 
        return result;
    }

    Tensor<T,2> backward(const Tensor<T,2>& g) override {
        Tensor<T,2> dZ = g; 
        
        auto s_it = temp.cbegin();
        for (auto dz_it = dZ.begin(); dz_it != dZ.end(); ++dz_it, ++s_it) {
            const T s = *s_it;
            *dz_it = *dz_it * (s * (T(1) - s));
        }
        return dZ;
    }
};
}

#endif //PROG3_NN_FINAL_PROJECT_V2025_01_ACTIVATION_H
