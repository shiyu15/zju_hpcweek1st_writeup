#ifndef QMATMUL_H
#define QMATMUL_H

#define GGML_COMMON_DECL_CPP
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <stdatomic.h>

#include "ggml-cpu.h"
#include "ggml-cpu-impl.h"
#include "ggml.h"
#include "quants.h"
#include "vec.h"
#include "pthread.h"

#ifdef __cplusplus
// restrict not standard in C++
#if defined(__GNUC__)
#define GGML_RESTRICT __restrict__
#elif defined(__clang__)
#define GGML_RESTRICT __restrict
#elif defined(_MSC_VER)
#define GGML_RESTRICT __restrict
#else
#define GGML_RESTRICT
#endif
#else
#if defined(_MSC_VER) && (__STDC_VERSION__ < 201112L)
#define GGML_RESTRICT __restrict
#else
#define GGML_RESTRICT restrict
#endif
#endif

#define GGML_CACHE_LINE  64

#if defined(__clang__) || defined(__GNUC__)
#define GGML_CACHE_ALIGN __attribute__((aligned(GGML_CACHE_LINE)))
#endif

typedef uint16_t ggml_fp16_t;
typedef pthread_cond_t     ggml_cond_t;
typedef pthread_mutex_t    ggml_mutex_t;
#define UNUSED GGML_UNUSED


// Threadpool def
struct ggml_threadpool {
    ggml_mutex_t mutex;       // mutex for cond.var
    ggml_cond_t  cond;        // cond.var for waiting for new work

    struct ggml_cgraph * cgraph;
    struct ggml_cplan  * cplan;

    // synchronization primitives
    atomic_int n_graph;       // incremented when there is work to be done (i.e each graph)
    atomic_int GGML_CACHE_ALIGN n_barrier;
    atomic_int GGML_CACHE_ALIGN n_barrier_passed;
    atomic_int GGML_CACHE_ALIGN current_chunk; // currently processing chunk during Mat_Mul, shared between all the threads.

    // these are atomic as an annotation for thread-sanitizer
    atomic_bool stop;         // Used for stopping the threadpool altogether
    atomic_bool pause;        // Used for pausing the threadpool or individual threads
    atomic_int abort;         // Used for aborting processing of a graph

    struct ggml_compute_state * workers;   // per thread state
    int          n_threads_max; // number of threads in the pool
    atomic_int   n_threads_cur; // number of threads used in the current graph

    int32_t      prio;        // Scheduling priority
    uint32_t     poll;        // Polling level (0 - no polling)

    enum ggml_status ec;
};

static inline float _GGML_CPU_FP16_TO_FP32(ggml_fp16_t h) {
    uint32_t sign = (h >> 15) & 1;
    uint32_t exp = (h >> 10) & 0x1f;
    uint32_t mant = h & 0x3ff;
    union { float f; uint32_t u; } out;

    if (exp == 0x1f) { // Infinity or NaN
        out.u = (sign << 31) | 0x7f800000 | (mant << 13);
    } else if (exp == 0) { // Denormalized number
        out.f = (sign ? -1.0f : 1.0f) * powf(2.0f, -14.0f) * (mant / 1024.0f);
    } else {
        exp += 127 - 15;
        mant <<= 13;
        out.u = (sign << 31) | (exp << 23) | mant;
    }
    return out.f;
}

static inline ggml_fp16_t _GGML_CPU_FP32_TO_FP16(float f) {
    union { float f; uint32_t u; } in = {f};
    uint32_t sign = (in.u >> 16) & 0x8000;
    int32_t exp = ((in.u >> 23) & 0xff) - 127;
    uint32_t mant = in.u & 0x7fffff;

    if (exp > 15) { return (ggml_fp16_t)(sign | (0x1f << 10)); } // 上溢
    if (exp < -14) { return (ggml_fp16_t)sign; } // 下溢

    return (ggml_fp16_t)(sign | ((exp + 15) << 10) | (mant >> 13));
}

void ggml_compute_forward_mul_mat_one_chunk(
    const struct ggml_compute_params* params, struct ggml_tensor* dst,
    const enum ggml_type type, const int64_t num_rows_per_vec_dot,
    const int64_t ir0_start, const int64_t ir0_end, const int64_t ir1_start,
    const int64_t ir1_end);

extern const struct ggml_type_traits_cpu type_traits_cpu[GGML_TYPE_COUNT];

void rvllm_vec_dot_q4_0_q8_0(int n, float* GGML_RESTRICT s, size_t bs,
                            const void* GGML_RESTRICT vx, size_t bx,
                            const void* GGML_RESTRICT vy, size_t by, int nrc);
#endif  // QMATMUL_H