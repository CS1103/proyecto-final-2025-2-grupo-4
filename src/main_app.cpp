#include <opencv2/opencv.hpp>
#include "../include/neural_network.h"
#include "keyboard_controller.h" 

utec::neural_network::Tensor<float, 2> mat_to_tensor(const cv::Mat& img) {
    utec::neural_network::Tensor<float, 2> t(1, 900);
    int idx = 0;
    for(int i=0; i<img.rows; ++i) 
        for(int j=0; j<img.cols; ++j) 
            t(0, idx++) = (float)img.at<uchar>(i,j) / 255.0f;
    return t;
}

int get_prediction(utec::neural_network::NeuralNetwork<float>& nn, const cv::Mat& img_crop) {
    cv::Mat img_small;
    cv::resize(img_crop, img_small, cv::Size(30, 30));
    
    auto output = nn.predict(mat_to_tensor(img_small));
    
    int best = 0;
    float max_p = 0.0f;
    for(int i=0; i<5; ++i) { 
        if (output(0, i) > max_p) {
            max_p = output(0, i);
            best = i;
        }
    }

    return (max_p > 0.5f) ? best : -1;
}

int main() {
    using namespace std;
    using namespace cv;

    utec::neural_network::NeuralNetwork<float> nn;
    auto l1 = new utec::neural_network::Dense<float>(900, 128, [](auto&){}, [](auto&){});
    auto l2 = new utec::neural_network::Dense<float>(128, 5, [](auto&){}, [](auto&){});
    
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

    cout << "CELESTE CONTROLLER: PANTALLA DIVIDIDA" << endl;

    while(true) {
        cap >> frame;
        if(frame.empty()) break;
        
        flip(frame, frame, 1);
        cvtColor(frame, gray, COLOR_BGR2GRAY);

        int mid_x = gray.cols / 2;
        Rect left_roi(0, 0, mid_x, gray.rows);      
        Rect right_roi(mid_x, 0, mid_x, gray.rows);

        Mat img_left = gray(left_roi);
        Mat img_right = gray(right_roi);

        // --- PREDICCIONES ---
        int p_left = get_prediction(nn, img_left);
        int p_right = get_prediction(nn, img_right);

        // --- MANO IZQUIERDA (MOVIMIENTO) ---

        kb.update_key(VK_RIGHT, (p_left == 1));
        kb.update_key(VK_LEFT,  (p_left == 2));
        kb.update_key(VK_UP,    (p_left == 3));
        kb.update_key(VK_DOWN,  (p_left == 4));

        // --- MANO DERECHA (ACCIONES) ---

        kb.update_key('X', (p_right == 0));
        kb.update_key('C', (p_right == 3));
        kb.update_key('Z', (p_right == 4));

        line(frame, Point(mid_x, 0), Point(mid_x, frame.rows), Scalar(0, 255, 255), 2);
        imshow("Celeste Split Screen", frame);
        if(waitKey(1) == 27) break;
    }
    return 0;
}