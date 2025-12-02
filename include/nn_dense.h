//
// Created by rudri on 10/11/2020.
//

#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_DENSE_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_DENSE_H
#include "nn_interfaces.h"
#include <fstream>
namespace utec::neural_network {

using utec::algebra::matrix_product;
using utec::algebra::transpose_2d;

template<typename T>
class Dense final : public ILayer<T> {
    private:
    Tensor<T,2> weights_;
    Tensor<T,2> bias_;
    Tensor<T,2> dW_;
    Tensor<T,2> db_;
    Tensor<T,2> temp;

    public:
    template<typename InitWFun, typename InitBFun>
    Dense(int in_f, int out_f, InitWFun init_w_fun, InitBFun init_b_fun) 
      : weights_((int)in_f, (int)out_f), 
        bias_(1, (int)out_f),
        dW_((int)in_f, (int)out_f), 
        db_(1, (int)out_f),
        temp(0, 0)
    {
        init_w_fun(weights_);
        init_b_fun(bias_);
    }

    Tensor<T,2> forward(const Tensor<T,2>& x) override {
        temp = x; 
        return matrix_product(x, weights_) + bias_;
    }


    Tensor<T,2> backward(const Tensor<T,2>& dZ) override {

        dW_ = matrix_product(transpose_2d(temp), dZ);

        db_.fill(T(0));
        for (int i = 0; i < dZ.shape()[0]; ++i) { // sobre n_batches
            for (int j = 0; j < dZ.shape()[1]; ++j) {
                db_(0, j) += dZ(i, j);
            }
        }
        
        // dL/dX para la capa anterior
        return matrix_product(dZ, transpose_2d(weights_));
    }

    void update_params(IOptimizer<T>& optimizer) override {
        optimizer.update(weights_, dW_);
        optimizer.update(bias_, db_);
    }
    // Agregar en epic3/nn_dense.h dentro de public:

    void save_params(const std::string& prefix) {
        // Guarda pesos
        std::ofstream f_w(prefix + "_weights.txt");
        for(auto val : weights_) f_w << val << " ";
        f_w.close();
        
        // Guarda sesgos (bias)
        std::ofstream f_b(prefix + "_bias.txt");
        for(auto val : bias_) f_b << val << " ";
        f_b.close();
    }

    void load_params(const std::string& prefix) {
        // Carga pesos
        std::ifstream f_w(prefix + "_weights.txt");
        for(auto& val : weights_) f_w >> val;
        f_w.close();

        // Carga sesgos
        std::ifstream f_b(prefix + "_bias.txt");
        for(auto& val : bias_) f_b >> val;
        f_b.close();
    }
};
}

#endif //PROG3_NN_FINAL_PROJECT_V2025_01_DENSE_H
