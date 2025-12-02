//
// Created by rudri on 10/11/2020.
//

#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_NEURAL_NETWORK_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_NEURAL_NETWORK_H

#include <vector>
#include <memory>
#include <iostream>
#include "nn_interfaces.h" 
#include "nn_optimizer.h"  
#include "nn_loss.h"       
#include "nn_dense.h"      
#include "nn_activation.h" 

namespace utec::neural_network {

template<typename T>
class NeuralNetwork {
private:
    std::vector<std::unique_ptr<ILayer<T>>> layers_;

public:
    void add_layer(std::unique_ptr<ILayer<T>> layer) {
        layers_.push_back(std::move(layer));
    }

    Tensor<T,2> predict(const Tensor<T,2>& X) {
        Tensor<T,2> current_output = X;
        for (auto& layer : layers_) {
            current_output = layer->forward(current_output);
        }
        return current_output;
    }

    template <template <typename ...> class LossType, 
              template <typename ...> class OptimizerType = SGD>
    void train( const Tensor<T,2>& X_train, 
                const Tensor<T,2>& Y_true, 
                const size_t epochs, 
                const size_t batch_size, 
                T learning_rate) 
    {
        OptimizerType<T> optimizer(learning_rate);
        
        for (size_t epoch = 0; epoch < epochs; ++epoch) {
            
            // preddiction
            Tensor<T,2> Y_pred = this->predict(X_train);
            
            // loss and grad
            LossType<T> loss_fn(Y_pred, Y_true);
            Tensor<T,2> loss_gradient = loss_fn.loss_gradient(); // dL/dY_pred
            
            // back propagation
            for (auto it = layers_.rbegin(); it != layers_.rend(); ++it) {
                loss_gradient = (*it)->backward(loss_gradient);
            }

            // change props
            optimizer.step(); // adam steps
            for (auto& layer : layers_) {
                layer->update_params(optimizer);
            }
            float progress = (epoch + 1) * 100.0f / epochs;
            if ((epoch + 1) % 10 == 0)   // cada 10 epochs
                std::cout << "\rProgreso: " << progress << "% completado" << std::flush;
        }
        std::cout << std::endl;
    }
};
}
#endif //PROG3_NN_FINAL_PROJECT_V2025_01_NEURAL_NETWORK_H
