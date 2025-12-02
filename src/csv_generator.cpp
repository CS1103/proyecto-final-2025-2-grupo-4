#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <map>
#include <vector>

using namespace std;
using namespace cv;
namespace fs = std::filesystem;

// 0: Puño, 1: Like, 2: Dislike, 3: Stop, 4: Peace
map<string, int> folder_map = {
    {"fist", 0},
    {"like", 1},
    {"dislike", 2},
    {"stop", 3},
    {"peace", 4}
};

const int MAX_IMAGES_PER_CLASS = 600; // Limite para no saturar

int main() {
    string dataset_root = "C:/Users/arana/Downloads/hagrid-classification-512p"; 
    
    ofstream csv("celeste_dataset.csv");
    if (!csv.is_open()) {
        cerr << "Error: No se pudo crear el archivo csv." << endl;
        return -1;
    }

    int total_count = 0;

    for (const auto& entry : fs::directory_iterator(dataset_root)) {
        if (entry.is_directory()) {
            string folder_name = entry.path().filename().string();
            
            if (folder_map.find(folder_name) == folder_map.end()) continue;

            int label = folder_map[folder_name];
            cout << "Procesando carpeta: " << folder_name << " (Clase " << label << ")..." << endl;

            int class_count = 0;
            for (const auto& img_entry : fs::directory_iterator(entry.path())) {
                if (class_count >= MAX_IMAGES_PER_CLASS) break;

                // Solo imagenes
                string ext = img_entry.path().extension().string();
                if (ext != ".jpg" && ext != ".jpeg" && ext != ".png") continue;

                Mat img = imread(img_entry.path().string(), IMREAD_GRAYSCALE);
                if (img.empty()) continue;

                // Resize a 30x30
                Mat img_small;
                resize(img, img_small, Size(30, 30));

                csv << label;
                for (int i = 0; i < img_small.rows; ++i) {
                    for (int j = 0; j < img_small.cols; ++j) {
                        csv << "," << (float)img_small.at<uchar>(i, j) / 255.0f;
                    }
                }
                csv << "\n";
                class_count++;
                total_count++;
            }
        }
    }
    cout << "Listo. Total imagenes: " << total_count << endl;
    return 0;
}