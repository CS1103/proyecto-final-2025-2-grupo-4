#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <map>
#include <vector>
#include <algorithm>
using namespace std;
using namespace cv;
namespace fs = std::filesystem;

map<string, int> folder_map = {
    {"fist", 0},
    {"like", 1},
    {"dislike", 2},
    {"stop", 3},
    {"peace", 4}
};

const int MAX_IMAGES_PER_CLASS = 600; 

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

            vector<string> image_paths;
            for (const auto& img_entry : fs::directory_iterator(entry.path())) {
                string ext = img_entry.path().extension().string();
                if (ext == ".jpg" || ext == ".jpeg" || ext == ".png") {
                    image_paths.push_back(img_entry.path().string());
                }
            }

            int class_count = 0;
            for (const auto& path : image_paths) {
                if (class_count >= MAX_IMAGES_PER_CLASS) break;

                Mat img = imread(path, IMREAD_GRAYSCALE); 
                if (img.empty()) continue;

                Mat small;
                resize(img, small, Size(30, 30));

                csv << label;
                for (int i = 0; i < small.rows; ++i) {
                    for (int j = 0; j < small.cols; ++j) {
                        csv << "," << (float)small.at<uchar>(i, j) / 255.0f;
                    }
                }
                csv << "\n";
                class_count++;
                total_count++;
            }
            cout << " -> Guardadas " << class_count << " imagenes." << endl;
        }
    }
    cout << "FINALIZADO. Total dataset: " << total_count << " filas." << endl;
    csv.close();
    return 0;
}