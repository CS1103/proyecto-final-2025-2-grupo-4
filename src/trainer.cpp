#include "../include/neural_network.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

using namespace utec::neural_network;
using namespace std;

// Función para cargar CSV y convertir etiqueta "3" en vector "0,0,0,1"
pair<Tensor<float,2>, Tensor<float,2>> load_dataset(string path, int samples) {
    Tensor<float, 2> X(samples, 900); // 30x30 pixels
    Tensor<float, 2> Y(samples, 4);   // 4 Clases (Neutral, Izq, Der, Salto)
    Y.fill(0.0f); // Llenar de ceros

    ifstream file(path);
    string line, val;
    int row = 0;

    while(getline(file, line) && row < samples) {
        stringstream ss(line);
        
        // 1. Leer etiqueta
        getline(ss, val, ',');
        int label = stoi(val);
        
        // One-Hot Encoding 
        if(label >= 0 && label < 4) {
            Y(row, label) = 1.0f;
        }

        // 2. Leer pixeles
        int col = 0;
        while(getline(ss, val, ',')) {
            X(row, col++) = stof(val);
        }
        row++;
    }
    return {X, Y};
}

int main() {
    int N_SAMPLES = 500; // sample number
    auto [X, Y] = load_dataset("celeste_dataset.csv", N_SAMPLES);

    NeuralNetwork<float> nn;

    // Architecture
    // Layer 1: 900 -> 64
    auto l1 = new Dense<float>(900, 64, [](auto& t){ t.fill(0.01); }, [](auto& t){ t.fill(0); });
    nn.add_layer(unique_ptr<ILayer<float>>(l1));
    nn.add_layer(make_unique<ReLU<float>>());

    // Output Layer: 64 -> 4 (Using Sigmoid for 0-1 probability per class)
    auto l2 = new Dense<float>(64, 4, [](auto& t){ t.fill(0.01); }, [](auto& t){ t.fill(0); });
    nn.add_layer(unique_ptr<ILayer<float>>(l2));
    nn.add_layer(make_unique<Sigmoid<float>>());

    cout << "Training..." << endl;
    // Usamos MSELoss. Funciona bien para One-Hot vectors simples.
    nn.train<MSELoss, Adam>(X, Y, 2000, 32, 0.001);

    l1->save_params("celeste_l1");
    l2->save_params("celeste_l2");
    
    cout << "Model saved!" << endl;
    return 0;
}