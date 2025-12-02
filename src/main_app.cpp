#include <opencv2/opencv.hpp>
#include "../include/neural_network.h"
#include "keyboard_controller.h" 

// Helper OpenCV -> Tensor
utec::neural_network::Tensor<float, 2> mat_to_tensor(const cv::Mat& img) {
    utec::neural_network::Tensor<float, 2> t(1, 900);
    int idx = 0;
    for(int i=0; i<img.rows; ++i) 
        for(int j=0; j<img.cols; ++j) 
            t(0, idx++) = (float)img.at<uchar>(i,j) / 255.0f;
    return t;
}

int main() {
    using namespace std;
    using namespace cv;

    // 1. Cargar Red
    utec::neural_network::NeuralNetwork<float> nn;
    auto l1 = new utec::neural_network::Dense<float>(900, 64, [](auto&){}, [](auto&){});
    auto l2 = new utec::neural_network::Dense<float>(64, 4, [](auto&){}, [](auto&){});
    
    // Asegúrate de que estos archivos existan (se crean al correr trainer.exe)
    l1->load_params("celeste_l1");
    l2->load_params("celeste_l2");

    nn.add_layer(unique_ptr<utec::neural_network::ILayer<float>>(l1));
    nn.add_layer(make_unique<utec::neural_network::ReLU<float>>());
    nn.add_layer(unique_ptr<utec::neural_network::ILayer<float>>(l2));
    nn.add_layer(make_unique<utec::neural_network::Sigmoid<float>>());

    // 2. Iniciar Sistema
    VideoCapture cap(0);
    KeyboardController kb;
    
    // CAMBIO IMPORTANTE: Renombramos 'small' a 'img_small'
    Mat frame, img_small, gray; 

    cout << "CONTROL NEURONAL ACTIVADO. Presiona ESC para salir." << endl;

    while(true) {
        cap >> frame;
        if(frame.empty()) break;

        // Procesar
        cvtColor(frame, gray, COLOR_BGR2GRAY);
        
        // CAMBIO AQUÍ: Usamos img_small
        resize(gray, img_small, Size(30, 30)); 
        
        // CAMBIO AQUÍ: Usamos img_small
        auto output = nn.predict(mat_to_tensor(img_small));

        // Buscar la clase ganadora (ArgMax)
        int best_class = 0;
        float max_prob = 0.0f;
        
        for(int i=0; i<4; ++i) {
            float val = output(0, i);
            if(val > max_prob) {
                max_prob = val;
                best_class = i;
            }
        }

        if(max_prob < 0.7f) best_class = -1; // -1 significa "ninguno seguro"

        // 3. Mapear a Teclas
        // Ajusta esto según tus clases (0=Forward, 1=Backwards, etc.)
        kb.update_key('C',      (best_class == 0)); 
        kb.update_key('X',      (best_class == 1)); 
        kb.update_key(VK_LEFT,  (best_class == 2)); 
        kb.update_key(VK_RIGHT, (best_class == 3)); 

        // UI
        string txt = "NEUTRAL";
        if(best_class == 0) txt = "SALTAR";
        if(best_class == 1) txt = "DASH";
        if(best_class == 2) txt = "IZQUIERDA";
        if(best_class == 3) txt = "DERECHA";

        putText(frame, txt, Point(30, 50), FONT_HERSHEY_SIMPLEX, 1.5, Scalar(0,255,0), 3);
        imshow("Celeste Controller", frame);
        
        if(waitKey(1) == 27) break;
    }
    return 0;
}