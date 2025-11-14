#include "test_case.h"
#include "csr_matrix.h"
#include "matrix_utils.h"
#include "spmm_opt.h"
#include "spmm_ref.h"
#include <algorithm>
#include <chrono>
#include <iostream>
#include <omp.h>

void flush_cache_all_cores(size_t flush_size_per_thread = 800 * 1024)
{
// 清理cache缓存
#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        std::vector<char> buffer(flush_size_per_thread, tid);
        volatile char sink = 0;
        for (size_t i = 0; i < buffer.size(); i += 64) {
            sink += buffer[i];
        }
        if (sink == 123)
            std::cout << "";
    }
}

void print_parameter(const int& m, const int& n, const int& k, const double& sparsity, const int& test_time)
{
    std::cout << "=== SpMM Performance Test ===" << std::endl;
    std::cout << "Matrix dimensions: " << m << " x " << k << " (sparse) * " << k << " x " << n << " (dense)";
    std::cout << "   Test iterations: " << test_time;
    std::cout << "  Sparsity ratio: " << sparsity << std::endl;
    return;
}
void run_benchmark_and_validate(
    CSRMatrix<float>* csr_matrix,
    const float* B,
    const float* C_ref,
    int n,
    int test_time)
{
    const int m = csr_matrix->rows;
    const int k = csr_matrix->cols;
    float* C_opt = (float*)calloc(m * n, sizeof(float));

    double min_time = 1e9;
    for (int i = 0; i < test_time; i++) {
        memset(C_opt, 0, m * n * sizeof(float));
        flush_cache_all_cores();
        auto iter_start = std::chrono::high_resolution_clock::now();
        spmm_cpu_opt(csr_matrix->row_ptr, csr_matrix->col_indices, csr_matrix->values, B, C_opt, m, n, k);
        auto iter_end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(iter_end - iter_start);
        min_time = std::min(duration.count() / 1e6, min_time);
    }

    std::cout << "CPU SpMM COST TIME: " << min_time << " ms";
    double gflops = (2.0 * csr_matrix->nnz * n * 1e-9) / (min_time / 1000.0);
    std::cout << "   CPU SpMM GFLOPS: " << gflops << std::endl;

    float max_diff = max_diff_twoMatrix_scaled(csr_matrix, B, n, C_opt, C_ref);
    bool is_correct = (max_diff < 0.02f);

    std::cout << (is_correct ? "correct √" : "false !!") << " max diff: " << max_diff << "\n";

    free(C_opt);
}

void test_spmm_cpu(const int m, const int n, const int k, const int test_time, const double sparsity)
{
    float* A = (float*)malloc(m * k * sizeof(float));
    float* B = (float*)malloc(k * n * sizeof(float));
    float* C_ref = (float*)calloc(m * n, sizeof(float));

    Gen_Matrix_sparsity(A, m, k, sparsity);
    Gen_Matrix(B, k, n);

    CSRMatrix<float>* csr_matrix = dense_to_csr(A, m, k);
    print_parameter(m, n, k, sparsity, test_time);
    spmm_cpu_ref(csr_matrix->row_ptr, csr_matrix->col_indices, csr_matrix->values, B, C_ref, m, n, k);

    run_benchmark_and_validate(csr_matrix, B, C_ref, n, test_time);

    free_csr_matrix(csr_matrix);
    free(A);
    free(B);
    free(C_ref);
}

void test_spmm_cpu_mtx(const std::string& filename, const int n, const int test_time)
{
    CSRMatrix<float>* csr_matrix = loadCSRFromMTX<float>(filename);
    if (!csr_matrix) {
        std::cerr << "Failed to load matrix from file: " << filename << std::endl;
        return;
    }

    const int m = csr_matrix->rows;
    const int k = csr_matrix->cols;
    float* B = (float*)malloc(k * n * sizeof(float));
    float* C_ref = (float*)calloc(m * n, sizeof(float));

    Gen_Matrix(B, k, n);
    print_parameter(m, n, k, 1.0 - (double)csr_matrix->nnz / csr_matrix->rows / csr_matrix->cols, test_time);
    spmm_cpu_ref(csr_matrix->row_ptr, csr_matrix->col_indices, csr_matrix->values, B, C_ref, m, n, k);

    run_benchmark_and_validate(csr_matrix, B, C_ref, n, test_time);

    free_csr_matrix(csr_matrix);
    free(B);
    free(C_ref);
}