#pragma once

void spmm_cpu_ref(const int* __restrict__ ptr, const int* __restrict__ idx, const float* __restrict__ val, const float* __restrict__ vin, float* __restrict__ vout, const int num_v, const int INFEATURE, const int k);
