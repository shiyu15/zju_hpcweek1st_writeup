#pragma once

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>
#include <numeric>
#include <omp.h>
#include <random>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

// CSR矩阵结构体模板
template <typename T>
struct CSRMatrix {
    T* values; // 非零元素值数组
    int* col_indices; // 列索引数组
    int* row_ptr; // 行指针数组
    int rows; // 矩阵行数
    int cols; // 矩阵列数
    int nnz; // 非零元素数量
};
// 矩阵访问宏，将二维索引转换为一维索引 (行优先存储)
#define MATRIX_INDEX(i, j, cols) ((i) * (cols) + (j))
// 普通矩阵转换为CSR格式
template <typename T>
CSRMatrix<T>* dense_to_csr(const T* dense_matrix, int rows, int cols)
{
    // 分配CSR矩阵内存
    CSRMatrix<T>* csr_matrix = (CSRMatrix<T>*)malloc(sizeof(CSRMatrix<T>));
    csr_matrix->row_ptr = (int*)malloc((rows + 1) * sizeof(int));
    csr_matrix->rows = rows;
    csr_matrix->cols = cols;
    // 首先计算非零元素数量，并填充row_ptr数组
    csr_matrix->row_ptr[0] = 0;
    int nnz = 0;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (dense_matrix[MATRIX_INDEX(i, j, cols)] != static_cast<T>(0)) {
                nnz++;
            }
        }
        csr_matrix->row_ptr[i + 1] = nnz;
    }
    csr_matrix->nnz = nnz;
    csr_matrix->values = (T*)malloc(nnz * sizeof(T));
    csr_matrix->col_indices = (int*)malloc(nnz * sizeof(int));

    // 填充其余CSR数据
    int idx = 0;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            T val = dense_matrix[MATRIX_INDEX(i, j, cols)];
            if (val != static_cast<T>(0)) {
                csr_matrix->values[idx] = val;
                csr_matrix->col_indices[idx] = j;
                idx++;
            }
        }
    }

    return csr_matrix;
}

template <typename T>
void Gen_Matrix_sparsity(T* a, int rows, int cols, double sparsity = 0.0)
{
    std::mt19937_64 gen(20250828 + time(NULL));
    std::normal_distribution<T> dist(0, 1);
    int total_elements = rows * cols;
    int no_zero_count = static_cast<int>(total_elements * (1.0 - sparsity));
    memset(a, 0, sizeof(T) * total_elements);
    // 创建索引数组并打乱
    std::vector<int> indices(total_elements);
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), gen);
    // 将前no_zero_count个位置设为非零
    for (int i = 0; i < no_zero_count; i++) {
        a[indices[i]] = dist(gen);
    }
}

// 释放CSR矩阵内存
template <typename T>
void free_csr_matrix(CSRMatrix<T>* csr_matrix)
{
    if (csr_matrix) {
        free(csr_matrix->values);
        free(csr_matrix->col_indices);
        free(csr_matrix->row_ptr);
        free(csr_matrix);
    }
}

// 打印CSR矩阵
template <typename T>
void print_csr_matrix(const CSRMatrix<T>* csr_matrix, const char* title = "CSR Matrix")
{
    std::cout << title << " (" << csr_matrix->rows << "x" << csr_matrix->cols
              << ", nnz=" << csr_matrix->nnz << "):\n";
    std::cout << "Row ptr: ";
    for (int i = 0; i <= csr_matrix->rows; i++) {
        std::cout << csr_matrix->row_ptr[i] << " ";
    }
    std::cout << "\n";
    std::cout << "Col indices: ";
    for (int i = 0; i < csr_matrix->nnz; i++) {
        std::cout << csr_matrix->col_indices[i] << " ";
    }
    std::cout << "\n";
    std::cout << "Values: ";
    for (int i = 0; i < csr_matrix->nnz; i++) {
        std::cout << csr_matrix->values[i] << " ";
    }
    std::cout << "\n";
}

// 计算矩阵稀疏度
template <typename T>
double calculate_sparsity(const T* matrix, int rows, int cols)
{
    int zero_count = 0;
    int total = rows * cols;
#pragma omp parallel for reduction(+ : zero_count) schedule(static, 128)
    for (int i = 0; i < total; i++) {
        if (matrix[i] == static_cast<T>(0)) {
            zero_count++;
        }
    }
    return static_cast<double>(zero_count) / total;
}

// 加载MTX文件并转换为CSR格式
template <typename T>
CSRMatrix<T>* loadCSRFromMTX(const std::string& filename)
{
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: cannot open .mtx file " << filename << std::endl;
        return nullptr;
    }

    std::string line;
    std::getline(file, line);

    std::stringstream ss_banner(line);
    std::string banner, mtx, format, field, symmetry;
    ss_banner >> banner >> mtx >> format >> field >> symmetry;
    std::transform(symmetry.begin(), symmetry.end(), symmetry.begin(), ::tolower);

    if (format != "coordinate") {
        std::cerr << "Error: Only 'coordinate' format is supported." << std::endl;
        return nullptr;
    }

    int rows, cols, stored_nnz;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '%')
            continue;
        std::stringstream ss_dims(line);
        ss_dims >> rows >> cols >> stored_nnz;
        break;
    }

    using COO_Entry = std::tuple<int, int, T>;
    std::vector<COO_Entry> coo_entries;
    coo_entries.reserve(stored_nnz * (symmetry == "general" ? 1 : 2));

    bool is_pattern = (field == "pattern");

    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '%')
            continue;
        std::stringstream ss_data(line);
        int r, c;
        T v = static_cast<T>(1);
        ss_data >> r >> c;
        if (!is_pattern) {
            ss_data >> v;
        }

        r--;
        c--;
        coo_entries.emplace_back(r, c, v);

        if (symmetry != "general" && r != c) {
            T symmetric_v = v;
            if (symmetry == "skew-symmetric") {
                symmetric_v = -v;
            }
            coo_entries.emplace_back(c, r, symmetric_v);
        }
    }
    file.close();

    std::sort(coo_entries.begin(), coo_entries.end());

    CSRMatrix<T>* matrix = (CSRMatrix<T>*)std::malloc(sizeof(CSRMatrix<T>));
    if (!matrix) {
        std::cerr << "Error: Failed to allocate memory for CSRMatrix struct." << std::endl;
        return nullptr;
    }

    matrix->rows = rows;
    matrix->cols = cols;
    matrix->nnz = static_cast<int>(coo_entries.size());
    matrix->values = (T*)std::malloc(matrix->nnz * sizeof(T));
    matrix->col_indices = (int*)std::malloc(matrix->nnz * sizeof(int));
    matrix->row_ptr = (int*)std::malloc((rows + 1) * sizeof(int));

    if (!matrix->values || !matrix->col_indices || !matrix->row_ptr) {
        std::cerr << "Error: Failed to allocate memory for CSR data arrays." << std::endl;
        std::free(matrix->values);
        std::free(matrix->col_indices);
        std::free(matrix->row_ptr);
        std::free(matrix);
        return nullptr;
    }

    std::fill(matrix->row_ptr, matrix->row_ptr + rows + 1, 0);

    for (size_t i = 0; i < coo_entries.size(); ++i) {
        const auto& [r, c, v] = coo_entries[i];
        matrix->values[i] = v;
        matrix->col_indices[i] = c;
        matrix->row_ptr[r + 1]++;
    }

    for (int i = 0; i < rows; ++i) {
        matrix->row_ptr[i + 1] += matrix->row_ptr[i];
    }

    return matrix;
}