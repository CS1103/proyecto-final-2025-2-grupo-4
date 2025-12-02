#include <opencv2/opencv.hpp>
#include "../include/neural_network.h"
#include "keyboard_controller.h"

using namespace cv;
using namespace utec::neural_network;
using namespace std;

// Helper OpenCV -> Tensor
Tensor<float, 2> mat_to_tensor(const Mat& img) {
    Tensor<float, 2> t(1, 900);
    int idx = 0;
    for(int i=0; i<img.rows; ++i) 
        for(int j=0; j<img.cols; ++j) 
            t(0, idx++) = (float)img.at<uchar>(i,j) / 255.0f;
    return t;
}

int main() {
    // 1. Cargar Red
    NeuralNetwork<float> nn;
    auto l1 = new Dense<float>(900, 64, [](auto&){}, [](auto&){});
    auto l2 = new Dense<float>(64, 4, [](auto&){}, [](auto&){});
    
    l1->load_params("celeste_l1");
    l2->load_params("celeste_l2");

    nn.add_layer(unique_ptr<ILayer<float>>(l1));
    nn.add_layer(make_unique<ReLU<float>>());
    nn.add_layer(unique_ptr<ILayer<float>>(l2));
    nn.add_layer(make_unique<Sigmoid<float>>());

    // 2. Iniciar Sistema
    VideoCapture cap(0);
    KeyboardController kb;
    Mat frame, small, gray;

    cout << "Presiona ESC para salir." << endl;

    while(true) {
        cap >> frame;
        if(frame.empty()) break;

        // Procesar
        cvtColor(frame, gray, COLOR_BGR2GRAY);
        resize(gray, small, Size(30, 30));
        
        // Predecir
        auto output = nn.predict(mat_to_tensor(small));

        // Buscar la clase ganadora 
        int best_class = 0;
        float max_prob = 0.0f;
        
        for(int i=0; i<4; ++i) {
            float val = output(0, i);
            if(val > max_prob) {
                max_prob = val;
                best_class = i;
            }
        }

        // Umbral de confianza (si no está seguro, no hace nada)
        if(max_prob < 0.7f) best_class = 0; 

        // 3. Mapear a Teclas (Configuración Celeste)
        // 0: Nada, 1: Izq, 2: Der, 3: Saltap
        kb.update_key(VK_LEFT,  (best_class == 1));
        kb.update_key(VK_RIGHT, (best_class == 2));
        kb.update_key('C',      (best_class == 3)); 

        // UI
        string txt = "NEUTRAL";
        if(best_class == 1) txt = "IZQUIERDA";
        if(best_class == 2) txt = "DERECHA";
        if(best_class == 3) txt = "SALTO";

        putText(frame, txt, Point(30, 50), FONT_HERSHEY_SIMPLEX, 1.5, Scalar(0,255,0), 3);
        imshow("Celeste ", frame);
        
        if(waitKey(1) == 27) break;
    }
    return 0;
}