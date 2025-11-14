#pragma GCC optimize("Ofast,no-stack-protector,fast-math")
#include <vector>
#include <limits>
#include <omp.h>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <new>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

template <typename T, std::size_t Alignment>
struct AlignedAllocator {
    using value_type = T;
    using pointer = T*;
    using const_pointer = const T*;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

    template <class U>
    struct rebind { using other = AlignedAllocator<U, Alignment>; };

    AlignedAllocator() noexcept = default;
    template <class U>
    AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}

    pointer allocate(size_type n) {
        void* raw = nullptr;
        if (posix_memalign(&raw, Alignment, n * sizeof(T)) != 0) {
            throw std::bad_alloc();
        }
        return static_cast<pointer>(raw);
    }

    void deallocate(pointer p, size_type) noexcept {
        std::free(p);
    }
};

template <typename T, std::size_t Alignment, typename U>
inline bool operator==(const AlignedAllocator<T, Alignment>&,
                       const AlignedAllocator<U, Alignment>&) noexcept {
    return true;
}

template <typename T, std::size_t Alignment, typename U>
inline bool operator!=(const AlignedAllocator<T, Alignment>& a,
                       const AlignedAllocator<U, Alignment>& b) noexcept {
    return !(a == b);
}

using AlignedUInt8Buffer = std::vector<uint8_t, AlignedAllocator<uint8_t, 32>>;

namespace py = pybind11;

// ---------- 工具：比较两个 char 网格是否完全一致 ----------
static inline bool equal_grid_char(const std::vector<std::vector<char>>& A,
                                   const std::vector<std::vector<char>>& B) {
    if (A.size() != B.size()) return false;
    for (size_t i = 0; i < A.size(); ++i) {
        if (A[i].size() != B[i].size()) return false;
        const char* pa = A[i].data();
        const char* pb = B[i].data();
        for (size_t j = 0; j < A[i].size(); ++j) {
            if (pa[j] != pb[j]) return false;
        }
    }
    return true;
}

#if defined(__GNUC__)
typedef unsigned char u8x16 __attribute__((vector_size(16), may_alias));
#else
#error "This code requires GNU vector extensions for u8x16."
#endif

static inline u8x16 load_u8x16(const uint8_t* ptr) {
    u8x16 v; __builtin_memcpy(&v, ptr, 16); return v;
}
static inline void store_u8x16(uint8_t* ptr, u8x16 v) {
    __builtin_memcpy(ptr, &v, 16);
}

static std::vector<std::vector<char>>
next_generation_char(const std::vector<std::vector<char>>& grid) {
    const int h = static_cast<int>(grid.size());
    const int w = (h > 0) ? static_cast<int>(grid[0].size()) : 0;
    if (h == 0 || w == 0) return {};

    // 1) pad 一圈 0
    const int H = h + 2, W = w + 2;
    const int stride = ((W + 31) / 32) * 32;
    AlignedUInt8Buffer padded(H * stride, 0);
    AlignedUInt8Buffer next_padded(H * stride, 0);

    for (int y = 0; y < h; ++y) {
        uint8_t* row = padded.data() + (y + 1) * stride;
        std::memcpy(row + 1, grid[y].data(), size_t(w));
    }

    // 2) 并行计算：沿 y 分成 16 个块，每个线程处理一段连续行

    int g_min_y = std::numeric_limits<int>::max();
    int g_min_x = std::numeric_limits<int>::max();
    int g_max_y = -1;
    int g_max_x = -1;

    const u8x16 VZERO = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
    const u8x16 VONE  = {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};
    const u8x16 V2    = {2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2};
    const u8x16 V3    = {3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3};

    auto L  = [](const uint8_t* p, int off) -> u8x16 { return load_u8x16(p + off); };
    auto LZ = [&](const uint8_t* p, int off) -> u8x16 { return p ? load_u8x16(p + off) : VZERO; };

    const int NT = 4;                                         // 线程数/分块数
    const int CH = (H + NT - 1) / NT;                          // 每块行数（最后一块可能较短）

    #pragma omp parallel for num_threads(NT) schedule(static,1) if(H*W >= 8192)
    for (int t = 0; t < NT; ++t) {
        const int y0 = t * CH;
        const int y1 = std::min(H, y0 + CH);
        if (y0 >= y1) continue;

        int local_min_y = std::numeric_limits<int>::max();
        int local_min_x = std::numeric_limits<int>::max();
        int local_max_y = -1;
        int local_max_x = -1;

        for (int y = y0; y < y1; ++y) {
            const uint8_t* prow_m1 = (y > 0)     ? padded.data() + (y - 1) * stride : nullptr;
            const uint8_t* prow    =               padded.data() +  y      * stride;
            const uint8_t* prow_p1 = (y + 1 < H) ? padded.data() + (y + 1) * stride : nullptr;
            uint8_t*       outrow  =               next_padded.data() +  y  * stride;

            // —— 标量：左边缘（避免 x-1 越界）——
            int x = 0;
            for (; x < 1 && x < W; ++x) {
                int s = 0;
                if (prow_m1) {
                    if (x > 0)      s += (int)prow_m1[x - 1];
                    s += (int)prow_m1[x];
                    if (x + 1 < W) s += (int)prow_m1[x + 1];
                }
                if (x > 0)      s += (int)prow[x - 1];
                if (x + 1 < W) s += (int)prow[x + 1];
                if (prow_p1) {
                    if (x > 0)      s += (int)prow_p1[x - 1];
                    s += (int)prow_p1[x];
                    if (x + 1 < W) s += (int)prow_p1[x + 1];
                }
                const int alive = (int)prow[x];
                const uint8_t nxt = (s == 3 || (s == 2 && alive == 1)) ? 1u : 0u;
                outrow[x] = nxt;

                if (nxt) {
                    if (y < local_min_y) local_min_y = y;
                    if (x < local_min_x) local_min_x = x;
                    if (y > local_max_y) local_max_y = y;
                    if (x > local_max_x) local_max_x = x;
                }
            }

            // —— SIMD：一次处理 16 列（可访问 x-1/x/x+1）——
            int vec_start = std::max(x, 1);                     // 至少从 x=1 开始
            int vec_end   = (W >= 1) ? W - 1 : 0;               // 留出最后一列给标量
            if (vec_end < vec_start) vec_end = vec_start;
            int vec_limit = vec_start + ((vec_end - vec_start) & ~15); // 对齐到 16 倍数

            for (x = vec_start; x + 15 < vec_limit; x += 16) {
                // 上一行
                u8x16 tL = LZ(prow_m1, x - 1);
                u8x16 tC = LZ(prow_m1, x    );
                u8x16 tR = LZ(prow_m1, x + 1);
                // 本行（左右）
                u8x16 mL = L (prow,    x - 1);

                // 当前细胞（0/1）
                u8x16 alive = L(prow, x);

                u8x16 mR = L (prow,    x + 1);
                // 下一行
                u8x16 bL = LZ(prow_p1, x - 1);
                u8x16 bC = LZ(prow_p1, x    );
                u8x16 bR = LZ(prow_p1, x + 1);

                // 8 邻居求和（0..8）
                u8x16 sum = tL + tC;
                sum = sum + tR;
                sum = sum + mL;
                sum = sum + mR;
                sum = sum + bL;
                sum = sum + bC;
                sum = sum + bR;

                

                // 规则：s==3 || (s==2 && alive==1)
                u8x16 eq3 = (sum == V3);                 // 0xFF / 0x00
                u8x16 eq2 = (sum == V2);                 // 0xFF / 0x00
                u8x16 alive_mask = (alive == VONE);      // 0xFF / 0x00
                u8x16 mask = eq3 | (eq2 & alive_mask);

                // 压成 1/0
                u8x16 nextv = mask & VONE;
                store_u8x16(outrow + x, nextv);

                // 粗 bbox：只要该 16 列块有非零
                uint64_t lanes[2];
                __builtin_memcpy(lanes, &nextv, 16);
                if ((lanes[0] | lanes[1]) != 0ull) {
                    if (y < local_min_y) local_min_y = y;
                    if (x < local_min_x) local_min_x = x;
                    if (y > local_max_y) local_max_y = y;
                    if (x + 15 > local_max_x) local_max_x = x + 15;
                }
            }

            // —— 标量：尾数 + 右边缘 —— 
            for (; x < W; ++x) {
                int s = 0;
                if (prow_m1) {
                    if (x > 0)      s += (int)prow_m1[x - 1];
                    s += (int)prow_m1[x];
                    if (x + 1 < W) s += (int)prow_m1[x + 1];
                }
                if (x > 0)      s += (int)prow[x - 1];
                if (x + 1 < W) s += (int)prow[x + 1];
                if (prow_p1) {
                    if (x > 0)      s += (int)prow_p1[x - 1];
                    s += (int)prow_p1[x];
                    if (x + 1 < W) s += (int)prow_p1[x + 1];
                }
                const int alive_i = (int)prow[x];
                const uint8_t nxt = (s == 3 || (s == 2 && alive_i == 1)) ? 1u : 0u;
                outrow[x] = nxt;

                if (nxt) {
                    if (y < local_min_y) local_min_y = y;
                    if (x < local_min_x) local_min_x = x;
                    if (y > local_max_y) local_max_y = y;
                    if (x > local_max_x) local_max_x = x;
                }
            }
        } // end for y

        // 合并到全局 bbox
        if (local_max_y >= 0) {
            #pragma omp critical
            {
                if (local_min_y < g_min_y) g_min_y = local_min_y;
                if (local_min_x < g_min_x) g_min_x = local_min_x;
                if (local_max_y > g_max_y) g_max_y = local_max_y;
                if (local_max_x > g_max_x) g_max_x = local_max_x;
            }
        }
    } // end parallel for

    // 3) 若全 0，返回空
    if (g_max_y < 0) return {};

    // 4) 裁剪（仍在 padded 平面）
    const int out_h = g_max_y - g_min_y + 1;
    const int out_w = g_max_x - g_min_x + 1;
    std::vector<std::vector<char>> out(out_h, std::vector<char>(out_w, 0));
    for (int y = 0; y < out_h; ++y) {
        const uint8_t* src_row = next_padded.data() + (g_min_y + y) * stride + g_min_x;
        std::memcpy(out[y].data(), src_row, size_t(out_w));
    }
    return out;
}

// ---------- 多步：入口 int 矩阵 -> 转 char；中间全部 char；出口再转回 int ----------
std::vector<std::vector<int>> expand_cpp(
    const std::vector<std::vector<int>>& initial_grid,
    int generations) {

    // int -> char（更小带宽，提升 cache locality）
    const int h0 = static_cast<int>(initial_grid.size());
    const int w0 = (h0 > 0) ? static_cast<int>(initial_grid[0].size()) : 0;

    std::vector<std::vector<char>> cur_char(h0, std::vector<char>(w0, 0));
    for (int y = 0; y < h0; ++y) {
        const std::vector<int>& src = initial_grid[y];
        char* dst = cur_char[y].data();
        for (int x = 0; x < w0; ++x) dst[x] = static_cast<char>(src[x] != 0);
    }

    for (int it = 0; it < generations; ++it) {
        auto next_char = next_generation_char(cur_char);
        if (equal_grid_char(next_char, cur_char)) {
            cur_char.swap(next_char);
            break; // 稳态提前退出
        }
        cur_char.swap(next_char);
        if (cur_char.empty()) break; // 全 0
    }

    // char -> int（返回值与接口保持一致）
    std::vector<std::vector<int>> out;
    out.resize(cur_char.size());
    for (size_t y = 0; y < cur_char.size(); ++y) {
        out[y].resize(cur_char[y].size());
        const char* src = cur_char[y].data();
        int* dst = out[y].data();
        for (size_t x = 0; x < cur_char[y].size(); ++x) dst[x] = (src[x] != 0) ? 1 : 0;
    }
    return out;
}


PYBIND11_MODULE(NG, m) {
    m.def("Expand_Cpp", &expand_cpp,
          "Simulate multiple generations of Conway's Game of Life and return all intermediate states",
          py::arg("initial_grid"), py::arg("generations"));
}