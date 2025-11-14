#pragma once

#include "csr_matrix.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <iostream>
#include <limits>
#include <numeric>
#include <omp.h>
#include <random>
#include <vector>

template <typename T>
void Gen_Matrix(T* a, int rows, int cols)
{

    const int num_threads = 8;
#pragma omp parallel num_threads(num_threads)
    {
        int tid = omp_get_thread_num();
        std::mt19937_64 gen(time(NULL) + tid);
        std::normal_distribution<T> dist(0, 2);
        int chunk_size = (rows * cols + num_threads - 1) / num_threads;
        for (int i = tid * chunk_size; i < (tid + 1) * chunk_size && i < rows * cols; i++) {
            a[i] = dist(gen);
        }
    }
}

template <typename T>
void Gen_Matrix2(T* a, int rows, int cols)
{
    std::mt19937_64 gen(time(NULL));
    std::normal_distribution<T> dist(0, 2);
    for (int i = 0; i < rows * cols; i++) {
        a[i] = dist(gen);
    }
}

// 打印普通矩阵
template <typename T>
void print_dense_matrix(const T* matrix, int rows, int cols, const char* title = "Dense Matrix")
{
    std::cout << title << " (" << rows << "x" << cols << "):\n";
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            std::cout << matrix[i * cols + j] << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

// 释放普通矩阵内存
template <typename T>
void free_dense_matrix(T* dense_matrix)
{
    if (dense_matrix) {
        free(dense_matrix);
    }
}

// 结果检验相关函数
template <typename T>
T infinity_norm_sparse(int rows, const int* row_ptr, const T* values)
{
    T max_row_sum = 0;
#pragma omp parallel for reduction(max : max_row_sum)
    for (int i = 0; i < rows; ++i) {
        T current_row_sum = 0;
        for (int j = row_ptr[i]; j < row_ptr[i + 1]; ++j) {
            current_row_sum += std::abs(values[j]);
        }
        max_row_sum = std::max(max_row_sum, current_row_sum);
    }
    return max_row_sum;
}

template <typename T>
T infinity_norm_dense(int rows, int cols, const T* matrix)
{
    T max_row_sum = 0;
#pragma omp parallel for reduction(max : max_row_sum)
    for (int i = 0; i < rows; ++i) {
        T current_row_sum = 0;
        for (int j = 0; j < cols; ++j) {
            current_row_sum += std::abs(matrix[i * cols + j]);
        }
        max_row_sum = std::max(max_row_sum, current_row_sum);
    }
    return max_row_sum;
}

template <typename T>
double max_diff_twoMatrix_scaled(
    const CSRMatrix<T>* A,
    const T* B,
    int B_cols,
    const T* C_ref,
    const T* C_opt)
{
    int rows_C = A->rows;
    int cols_C = B_cols;
    int rows_B = A->cols;

    // 1. 计算 C_ref 和 C_opt 之间的最大绝对误差
    T max_abs_diff = 0;
#pragma omp parallel for reduction(max : max_abs_diff)
    for (int i = 0; i < rows_C * cols_C; i++) {
        max_abs_diff = std::max(max_abs_diff, std::abs(C_ref[i] - C_opt[i]));
    }

    // 2. 计算输入矩阵的无穷范数
    T norm_A = infinity_norm_sparse(A->rows, A->row_ptr, A->values);
    T norm_B = infinity_norm_dense(rows_B, B_cols, B);
    T epsilon = std::numeric_limits<T>::epsilon();

    // 3. 计算缩放残差
    double scaled_residual = 0.0;
    T denominator = norm_A * norm_B * epsilon;

    // 避免除以零或极小的数
    if (denominator < 1e-20) {
        // 如果输入矩阵范数为零，理论上输出也应为零，此时任何非零误差都是无穷大
        return (max_abs_diff > 0) ? std::numeric_limits<double>::infinity() : 0.0;
    }

    scaled_residual = static_cast<double>(max_abs_diff) / denominator;

    return scaled_residual;
}

