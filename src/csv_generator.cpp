#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <map>

using namespace std;
using namespace cv;
namespace fs = std::filesystem;


map<string, int> folder_map = {
    {"forward", 0}, 
    {"backwards", 1},
    {"left", 2},
    {"right", 3}
};

int main() {
    string dataset_root = "../dataset_green"; 
    
    ofstream csv("dataset_juego.csv");
    int count = 0;

    for (const auto& entry : fs::directory_iterator(dataset_root)) {
        if (entry.is_directory()) {
            string folder_name = entry.path().filename().string();
            
            if (folder_map.find(folder_name) == folder_map.end()) continue;

            int label = folder_map[folder_name];
            cout << "Procesando " << folder_name << " como clase " << label << "..." << endl;

            for (const auto& img_entry : fs::directory_iterator(entry.path())) {
                string path = img_entry.path().string();
                Mat img = imread(path, IMREAD_GRAYSCALE);
                
                if (img.empty()) continue;

                // Redimensionar a 30x30 (900 neuronas)
                Mat small;
                resize(img, small, Size(30, 30));

                csv << label;
                for (int i = 0; i < small.rows; ++i) {
                    for (int j = 0; j < small.cols; ++j) {
                        csv << "," << (float)small.at<uchar>(i, j) / 255.0f;
                    }
                }
                csv << "\n";
                count++;
            }
        }
    }
    cout << "Listo! Generadas " << count << " filas en dataset_juego.csv" << endl;
    return 0;
}