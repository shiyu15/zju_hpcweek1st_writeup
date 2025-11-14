#include <H5Cpp.h>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>
#include <fstream>
#include <algorithm>
#include <chrono>
#include "fbp.h"

namespace fs = std::filesystem;

static std::vector<hsize_t> get_shape(const H5::DataSet& ds) {
    H5::DataSpace sp = ds.getSpace();
    int nd = sp.getSimpleExtentNdims();
    std::vector<hsize_t> dims(nd);
    sp.getSimpleExtentDims(dims.data(), nullptr);
    return dims;
}

static void read_dataset_float32(const H5::DataSet& ds, std::vector<float>& out, std::vector<hsize_t>& shape) {
    shape = get_shape(ds);
    H5::DataType t = ds.getDataType();
    out.resize(std::accumulate(shape.begin(), shape.end(), hsize_t(1), std::multiplies<hsize_t>()));
    
    if (t == H5::PredType::NATIVE_FLOAT) {
        ds.read(out.data(), H5::PredType::NATIVE_FLOAT);
    } else if (t == H5::PredType::NATIVE_DOUBLE) {
        std::vector<double> tmp(out.size());
        ds.read(tmp.data(), H5::PredType::NATIVE_DOUBLE);
        for (size_t i = 0; i < out.size(); ++i) out[i] = static_cast<float>(tmp[i]);
    } else {
        throw std::runtime_error("Unsupported data type");
    }
}

static std::vector<float> generate_angles(int n_angles) {
    std::vector<float> angles(n_angles);
    for (int i = 0; i < n_angles; ++i) {
        angles[i] = 180.0f * i / n_angles;
    }
    return angles;
}

static void save_png(const fs::path& path, const float* img, int h, int w) {
    fs::create_directories(path.parent_path());
    
    float mn = img[0], mx = img[0];
    size_t total = size_t(h) * w;
    for (size_t i = 0; i < total; ++i) { 
        mn = std::min(mn, img[i]); 
        mx = std::max(mx, img[i]); 
    }
    float den = (mx > mn) ? (mx - mn) : 1.0f;
    
    cv::Mat mat(h, w, CV_8UC1);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            float v = (img[y * w + x] - mn) / den;
            mat.at<uint8_t>(h - 1 - y, x) = std::max(0, std::min(255, int(std::round(v * 255.0f))));
        }
    }
    
    cv::imwrite(path.string(), mat);
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <hdf5_file>\n";
        return 1;
    }

    try {
        // ============================================================
        // Load HDF5 sinogram data
        // ============================================================
        
        H5::H5File f(argv[1], H5F_ACC_RDONLY);
        std::cout << "HDF5 file: " << f.getFileName() << "\n";
        
        std::string dset_path = "/data";
        if (!f.nameExists(dset_path)) {
            std::cerr << "Dataset " << dset_path << " not found in file.\n";
            return 1;
        }
        
        std::cout << "Using dataset: " << dset_path << "\n";
        H5::DataSet ds = f.openDataSet(dset_path);
        
        // Get dataset dimensions
        std::vector<hsize_t> shape = get_shape(ds);
        if (shape.size() != 3) {
            std::cout << "Unsupported data dimensions: " << shape.size() << "D (expected 3D)\n";
            return 1;
        }
        
        int n_slices = int(shape[0]);
        int n_angles = int(shape[1]);
        int n_det = int(shape[2]);
        std::cout << "Data shape: [" << n_slices << " slices, " 
                  << n_angles << " angles, " << n_det << " detectors]\n";
        
        size_t slice_size = size_t(n_angles) * n_det;
        size_t recon_size = size_t(n_det) * n_det;
        size_t total_sino_size = n_slices * slice_size;
        size_t total_recon_size = n_slices * recon_size;
        
        // ============================================================
        // Allocate memory buffers
        // ============================================================
        
        std::vector<float> sino_buffer(total_sino_size);
        std::vector<float> recon_buffer(total_recon_size);
        
        // ============================================================
        // Read sinogram from HDF5
        // ============================================================
        
        std::cout << "Reading sinogram data from HDF5...\n";
        H5::DataType t = ds.getDataType();
        
        if (t == H5::PredType::NATIVE_FLOAT) {
            ds.read(sino_buffer.data(), H5::PredType::NATIVE_FLOAT);
        } else if (t == H5::PredType::NATIVE_DOUBLE) {
            // Read as double, then convert to float
            std::vector<double> tmp(total_sino_size);
            ds.read(tmp.data(), H5::PredType::NATIVE_DOUBLE);
            
            #pragma omp parallel for
            for (size_t i = 0; i < total_sino_size; ++i) {
                sino_buffer[i] = static_cast<float>(tmp[i]);
            }
        } else {
            throw std::runtime_error("Unsupported data type");
        }
        
        // Generate uniformly spaced angles [0, 180) degrees
        auto angles = generate_angles(n_angles);
        
        // ============================================================
        // Perform FBP reconstruction
        // ============================================================
        
        std::cout << "\nStarting FBP reconstruction...\n";
        
        // Start timing
        auto t_start = std::chrono::high_resolution_clock::now();
        
        fbp_reconstruct_3d(
            sino_buffer.data(),   // Input: will be filtered in-place
            recon_buffer.data(),  // Output: reconstructed volume
            n_slices,
            n_angles,
            n_det,
            angles
        );
        
        // End timing
        auto t_end = std::chrono::high_resolution_clock::now();
        auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();
        auto duration_s = duration_ms / 1000.0;
        
        std::cout << "FBP reconstruction completed in " << duration_s << " seconds\n";
        std::cout << "Average time per slice: " << (duration_ms / double(n_slices)) << " ms\n";
        
        // ============================================================
        // Save results as PNG images
        // ============================================================
        
        std::cout << "\nSaving results...\n";
        fs::create_directories("recon_out");
        
        for (int i = 0; i < n_slices; ++i) {
            char filename[64];
            
            // Save reconstructed image
            const float* recon_ptr = recon_buffer.data() + i * recon_size;
            snprintf(filename, sizeof(filename), "recon_out/recon_%03d.png", i);
            save_png(filename, recon_ptr, n_det, n_det);
            
        }
        
        std::cout << "All results saved to recon_out/\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
