#include "../include/neural_network.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <random>
#include <ctime>

using namespace utec::neural_network;
using namespace std;

void init_random(Tensor<float, 2>& t) {
    static std::default_random_engine gen((unsigned int)time(0));
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
    for(auto& val : t) val = dist(gen);
}

pair<Tensor<float,2>, Tensor<float,2>> load_dataset(string path) {
    ifstream count_file(path);
    if (!count_file.is_open()) {
        cerr << "ERROR: No se encuentra " << path << ". Ejecuta primero csv_generator." << endl;
        exit(1);
    }
    int samples = 0;
    string unused;
    while (getline(count_file, unused)) if(!unused.empty()) ++samples;
    count_file.close();

    cout << "Detectadas " << samples << " muestras." << endl;

    Tensor<float, 2> X(samples, 900); 
    Tensor<float, 2> Y(samples, 5); 
    Y.fill(0.0f); 

    ifstream file(path);
    string line, val;
    int row = 0;

    while(getline(file, line) && row < samples) {
        if (line.empty()) continue;
        stringstream ss(line);
        
        if (!getline(ss, val, ',')) continue;
        try {
            int label = stoi(val);
            if(label >= 0 && label < 5) Y(row, label) = 1.0f;
        } catch(...) { continue; }

        int col = 0;
        while(getline(ss, val, ',') && col < 900) {
            try { X(row, col++) = stof(val); } catch(...) {}
        }
        row++;
    }
    return {X, Y};
}

int main() {
    auto [X, Y] = load_dataset("celeste_dataset.csv");

    NeuralNetwork<float> nn;

    // Usamos init_random en los pesos
    auto l1 = new Dense<float>(900, 64, init_random, [](auto& t){ t.fill(0); });
    nn.add_layer(unique_ptr<ILayer<float>>(l1));
    nn.add_layer(make_unique<ReLU<float>>());

    auto l2 = new Dense<float>(64, 5, init_random, [](auto& t){ t.fill(0); });
    nn.add_layer(unique_ptr<ILayer<float>>(l2));
    nn.add_layer(make_unique<Sigmoid<float>>());

    cout << "Iniciando entrenamiento (Modo Release recomendado)..." << endl;
    
    // 3000 épocas para asegurar convergencia
    nn.train<MSELoss, Adam>(X, Y, 5000, 128, 0.001);

    cout << "Guardando pesos..." << endl;
    l1->save_params("celeste_l1");
    l2->save_params("celeste_l2");
    
    cout << "Listo!" << endl;
    return 0;
}