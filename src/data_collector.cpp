#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>

using namespace cv;
using namespace std;

void save_row(ofstream& f, const Mat& img, int label) {
    f << label;
    for(int i=0; i<img.rows; ++i) {
        for(int j=0; j<img.cols; ++j) {
            // Normalizar a 0-1 
            f << "," << (float)img.at<uchar>(i,j)/255.0f;
        }
    }
    f << "\n";
    cout << "Guardado sample clase: " << label << endl;
}

int main() {
    VideoCapture cap(0);
    if(!cap.isOpened()) return -1;

    // gardado de datos
    ofstream file("celeste_dataset.csv", ios::app); 
    Mat frame, small, gray;

    cout << "--- CONTROLES DE GRABACION ---" << endl;
    cout << "[0] Neutral (Soltar todo)" << endl;
    cout << "[1] Izquierda (Inclinate izq)" << endl;
    cout << "[2] Derecha (Inclinate der)" << endl;
    cout << "[3] Salto (Manos arriba o boca abierta)" << endl;
    cout << "ESC para salir" << endl;

    while(true) {
        cap >> frame;
        if(frame.empty()) break;

        // Preprocesamiento 
        cvtColor(frame, gray, COLOR_BGR2GRAY);
        resize(gray, small, Size(30, 30)); // 900 neuronas entrada

        imshow("Camara", frame);
        imshow("Red ve esto", small);

        char k = (char)waitKey(30);
        if(k == 27) break;

        // Guardar según tecla
        if(k == '0') save_row(file, small, 0);
        if(k == '1') save_row(file, small, 1);
        if(k == '2') save_row(file, small, 2);
        if(k == '3') save_row(file, small, 3);
    }
    return 0;
}