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

// Función auxiliar para obtener la clase ganadora
int get_prediction(utec::neural_network::NeuralNetwork<float>& nn, const cv::Mat& img_crop) {
    cv::Mat small_img;
    cv::resize(img_crop, small_img, cv::Size(30, 30));
    
    auto output = nn.predict(mat_to_tensor(small_img));
    
    int best = 0;
    float max_p = 0.0f;
    for(int i=0; i<5; ++i) { // 5 Clases
        if (output(0, i) > max_p) {
            max_p = output(0, i);
            best = i;
        }
    }
    // Si la confianza es baja (< 0.6), retornamos -1 (ninguna acción)
    return (max_p > 0.6f) ? best : -1;
}

int main() {
    using namespace std;
    using namespace cv;

    // 1. Cargar Red (Arquitectura 900 -> 64 -> 5)
    utec::neural_network::NeuralNetwork<float> nn;
    auto l1 = new utec::neural_network::Dense<float>(900, 64, [](auto&){}, [](auto&){});
    auto l2 = new utec::neural_network::Dense<float>(64, 5, [](auto&){}, [](auto&){});
    
    l1->load_params("celeste_l1");
    l2->load_params("celeste_l2");

    nn.add_layer(unique_ptr<utec::neural_network::ILayer<float>>(l1));
    nn.add_layer(make_unique<utec::neural_network::ReLU<float>>());
    nn.add_layer(unique_ptr<utec::neural_network::ILayer<float>>(l2));
    nn.add_layer(make_unique<utec::neural_network::Sigmoid<float>>());

    // 2. Iniciar
    VideoCapture cap(0);
    KeyboardController kb;
    Mat frame, gray;

    cout << "SISTEMA DE DOBLE MANO INICIADO." << endl;

    while(true) {
        cap >> frame;
        if(frame.empty()) break;
        
        flip(frame, frame, 1);

        cvtColor(frame, gray, COLOR_BGR2GRAY);

        int mid_x = gray.cols / 2;
=        Rect left_roi(0, 0, mid_x, gray.rows);
        Rect right_roi(mid_x, 0, mid_x, gray.rows);

        Mat img_left = gray(left_roi);
        Mat img_right = gray(right_roi);

        // --- PREDICCIONES ---
        int p_left = get_prediction(nn, img_left);
        int p_right = get_prediction(nn, img_right);


        kb.update_key(VK_RIGHT, (p_left == 1));
        kb.update_key(VK_LEFT,  (p_left == 2));
        kb.update_key(VK_UP,    (p_left == 3));
        kb.update_key(VK_DOWN,  (p_left == 4));


        kb.update_key('X', (p_right == 0));
        kb.update_key('C', (p_right == 3));
        kb.update_key('Z', (p_right == 4));


        // Dibujar línea divisoria
        line(frame, Point(mid_x, 0), Point(mid_x, frame.rows), Scalar(0, 255, 255), 2);
        
        string t_izq = "---";
        if(p_left == 0) t_izq = "NEUTRO";
        if(p_left == 1) t_izq = "DER ->";
        if(p_left == 2) t_izq = "<- IZQ";
        if(p_left == 3) t_izq = "^ ARR";
        if(p_left == 4) t_izq = "v ABA";
        putText(frame, t_izq, Point(50, 50), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 2);

        string t_der = "---";
        if(p_right == 0) t_der = "DASH (X)";
        if(p_right == 3) t_der = "SALTAR (C)";
        if(p_right == 4) t_der = "AGARRAR (Z)";
        putText(frame, t_der, Point(mid_x + 50, 50), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 2);

        imshow("Celeste Controller (Split)", frame);
        if(waitKey(1) == 27) break;
    }
    return 0;
}