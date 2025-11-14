#pragma once

#include <string>

void test_spmm_cpu(const int m, const int n, const int k, const int test_time, const double sparsity);

void test_spmm_cpu_mtx(const std::string& filename, const int n, const int test_time);