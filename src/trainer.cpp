#include "../include/neural_network.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <random> // NECESARIO PARA ALEATORIEDAD
#include <ctime>

using namespace utec::neural_network;
using namespace std;

// --- FUNCIÓN DE INICIALIZACIÓN ALEATORIA ---
// Llena los pesos con valores al azar entre -0.1 y 0.1
void init_random(Tensor<float, 2>& t) {
    static std::default_random_engine gen((unsigned int)time(0));
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
    
    for(auto& val : t) {
        val = dist(gen);
    }
}
// -------------------------------------------

// Función de carga segura (la que te di antes)
pair<Tensor<float,2>, Tensor<float,2>> load_dataset(string path, int samples) {
    Tensor<float, 2> X(samples, 900); 
    Tensor<float, 2> Y(samples, 4);   
    Y.fill(0.0f); 

    ifstream file(path);
    if (!file.is_open()) {
        cout << "ERROR CRITICO: No se encuentra " << path << endl;
        exit(1);
    }

    string line, val;
    int row = 0;
    while(getline(file, line) && row < samples) {
        if (line.empty() || line.size() < 10) continue; // Saltar líneas vacías
        stringstream ss(line);
        
        // Etiqueta
        if (!getline(ss, val, ',')) continue;
        try {
            int label = stoi(val);
            if(label >= 0 && label < 4) Y(row, label) = 1.0f;
        } catch(...) { continue; }

        // Pixeles
        int col = 0;
        while(getline(ss, val, ',') && col < 900) {
            try { X(row, col++) = stof(val); } catch(...) { }
        }
        row++;
    }
    cout << "Dataset cargado: " << row << " filas validas." << endl;
    return {X, Y};
}

int main() {
    // IMPORTANTE: Asegúrate de que este número sea igual o menor 
    // a las líneas reales que tiene tu celeste_dataset.csv
    int N_SAMPLES = 150; 
    
    cout << "Cargando datos..." << endl;
    auto [X, Y] = load_dataset("celeste_dataset.csv", N_SAMPLES);

    NeuralNetwork<float> nn;

    // --- ARQUITECTURA CORREGIDA ---
    // Usamos 'init_random' en lugar de 't.fill'
    
    // Capa 1: 900 -> 64
    auto l1 = new Dense<float>(900, 64, init_random, [](auto& t){ t.fill(0); });
    nn.add_layer(unique_ptr<ILayer<float>>(l1));
    nn.add_layer(make_unique<ReLU<float>>());

    // Capa 2: 64 -> 4
    auto l2 = new Dense<float>(64, 4, init_random, [](auto& t){ t.fill(0); });
    nn.add_layer(unique_ptr<ILayer<float>>(l2));
    nn.add_layer(make_unique<Sigmoid<float>>());

    cout << "Entrenando (esto puede tardar unos segundos)..." << endl;
    
    // Subimos un poco las épocas para asegurar aprendizaje
    nn.train<MSELoss, Adam>(X, Y, 3000, 32, 0.001);

    cout << "Guardando pesos con variacion..." << endl;
    l1->save_params("celeste_l1");
    l2->save_params("celeste_l2");
    
    cout << "Modelo guardado correctamente!" << endl;
    return 0;
}