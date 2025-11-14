#pragma once

void spmm_cpu_opt(const int* __restrict__ ptr, const int* __restrict__ idx, const float* __restrict__ val, const float* __restrict__ vin, float* __restrict__ vout, int num_v, int INFEATURE, int k);
