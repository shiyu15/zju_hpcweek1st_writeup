#include "qmatmul.h"

void ggml_compute_forward_mul_mat(const struct ggml_compute_params* params,
                                  struct ggml_tensor* dst) {
  const struct ggml_tensor* src0 = dst->src[0];
  const struct ggml_tensor* src1 = dst->src[1];

  GGML_TENSOR_BINARY_OP_LOCALS

  const int ith = params->ith;
  const int nth = params->nth;

  enum ggml_type const vec_dot_type = type_traits_cpu[src0->type].vec_dot_type;
  ggml_from_float_t const from_float = type_traits_cpu[vec_dot_type].from_float;
  int64_t const vec_dot_num_rows = type_traits_cpu[src0->type].nrows;

  GGML_ASSERT(ne0 == ne01);
  GGML_ASSERT(ne1 == ne11);
  GGML_ASSERT(ne2 == ne12);
  GGML_ASSERT(ne3 == ne13);

  // we don't support permuted src0 or src1
  GGML_ASSERT(nb00 == ggml_type_size(src0->type));
  GGML_ASSERT(nb10 == ggml_type_size(src1->type));

  // dst cannot be transposed or permuted
  GGML_ASSERT(nb0 == sizeof(float));
  GGML_ASSERT(nb0 <= nb1);
  GGML_ASSERT(nb1 <= nb2);
  GGML_ASSERT(nb2 <= nb3);

  // nb01 >= nb00 - src0 is not transposed
  //   compute by src0 rows

  // TODO: extract to "extra_op"
#if GGML_USE_LLAMAFILE
  // broadcast factors
  const int64_t r2 = ne12 / ne02;
  const int64_t r3 = ne13 / ne03;

  const bool src1_cont = ggml_is_contiguous(src1);

  if (src1_cont) {
    for (int64_t i13 = 0; i13 < ne13; i13++)
      for (int64_t i12 = 0; i12 < ne12; i12++)
        if (!llamafile_sgemm(
                params, ne01, ne11, ne00 / ggml_blck_size(src0->type),
                (const char*)src0->data + i12 / r2 * nb02 + i13 / r3 * nb03,
                nb01 / ggml_type_size(src0->type),
                (const char*)src1->data + i12 * nb12 + i13 * nb13,
                nb11 / ggml_type_size(src1->type),
                (char*)dst->data + i12 * nb2 + i13 * nb3,
                nb1 / ggml_type_size(dst->type), src0->type, src1->type,
                dst->type))
          goto UseGgmlGemm1;
    return;
  }
UseGgmlGemm1:;
#endif

  if (src1->type != vec_dot_type) {
    char* wdata = params->wdata;

    const size_t nbw0 = ggml_type_size(vec_dot_type);
    const size_t nbw1 = ggml_row_size(vec_dot_type, ne10);
    const size_t nbw2 = nbw1 * ne11;
    const size_t nbw3 = nbw2 * ne12;

    assert(params->wsize >= ne13 * nbw3);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);

#if 0
        for (int64_t i13 = 0; i13 < ne13; ++i13) {
            for (int64_t i12 = 0; i12 < ne12; ++i12) {
                for (int64_t i11 = ith; i11 < ne11; i11 += nth) {
                    from_float((float *)((char *) src1->data + i13*nb13 + i12*nb12 + i11*nb11),
                               (void *)               (wdata + i13*nbw3 + i12*nbw2 + i11*nbw1),
                                ne10);
                }
            }
        }
#else
    for (int64_t i13 = 0; i13 < ne13; ++i13) {
      for (int64_t i12 = 0; i12 < ne12; ++i12) {
        for (int64_t i11 = 0; i11 < ne11; ++i11) {
          size_t bs = ggml_blck_size(vec_dot_type);
          int64_t ne10_block_start = (ith * ne10 / bs) / nth;
          int64_t ne10_block_end = ((ith + 1) * ne10 / bs) / nth;
          from_float((float*)((char*)src1->data + i13 * nb13 + i12 * nb12 +
                              i11 * nb11 + ne10_block_start * bs * nb10),
                     (void*)(wdata + i13 * nbw3 + i12 * nbw2 + i11 * nbw1 +
                             ne10_block_start * nbw0),
                     (ne10_block_end - ne10_block_start) * bs);
        }
      }
    }
#endif
  }

  if (ith == 0) {
    // Every thread starts at ith, so the first unprocessed chunk is nth.  This
    // save a bit of coordination right at the start.
    atomic_store_explicit(&params->threadpool->current_chunk, nth,
                          memory_order_relaxed);
  }

  ggml_barrier(params->threadpool);

#if GGML_USE_LLAMAFILE
  if (src1->type != vec_dot_type) {
    const void* wdata =
        (src1->type == vec_dot_type) ? src1->data : params->wdata;
    const size_t row_size = ggml_row_size(vec_dot_type, ne10);

    for (int64_t i13 = 0; i13 < ne13; i13++)
      for (int64_t i12 = 0; i12 < ne12; i12++)
        if (!llamafile_sgemm(
                params, ne01, ne11, ne00 / ggml_blck_size(src0->type),
                (const char*)src0->data + i12 / r2 * nb02 + i13 / r3 * nb03,
                nb01 / ggml_type_size(src0->type),
                (const char*)wdata +
                    (i12 * ne11 + i13 * ne12 * ne11) * row_size,
                row_size / ggml_type_size(vec_dot_type),
                (char*)dst->data + i12 * nb2 + i13 * nb3,
                nb1 / ggml_type_size(dst->type), src0->type, vec_dot_type,
                dst->type))
          goto UseGgmlGemm2;
    return;
  }
UseGgmlGemm2:;
#endif

  // This is the size of the first dimension of the result, so we can iterate
  // that way. (see the ASSERT above, these are the same numbers)
  const int64_t nr0 = ne0;

  // This is the size of the rest of the dimensions of the result
  const int64_t nr1 = ne1 * ne2 * ne3;

  // Now select a reasonable chunk size.
  int chunk_size = 16;

  // We need to step up the size if it's small
  if (nr0 == 1 || nr1 == 1) {
    chunk_size = 64;
  }

  // distribute the work across the inner or outer loop based on which one is
  // larger The number of chunks in the 0/1 dim. CEIL(nr0/chunk_size)
  int64_t nchunk0 = (nr0 + chunk_size - 1) / chunk_size;
  int64_t nchunk1 = (nr1 + chunk_size - 1) / chunk_size;

  // If the chunking is poor for the number of threads on this setup, scrap the
  // whole plan.  Re-chunk it by thread.
  //   Also, chunking by thread was measured to have perform better on NUMA
  //   systems.  See https://github.com/ggml-org/llama.cpp/pull/6915 In theory,
  //   chunking should be just as useful on NUMA and non NUMA systems, but
  //   testing disagreed with that.
  if (nchunk0 * nchunk1 < nth * 4 || ggml_is_numa()) {
    // distribute the thread work across the inner or outer loop based on which
    // one is larger
    nchunk0 = nr0 > nr1 ? nth : 1;  // parallelize by src0 rows
    nchunk1 = nr0 > nr1 ? 1 : nth;  // parallelize by src1 rows
  }

  // The number of elements in each chunk
  const int64_t dr0 = (nr0 + nchunk0 - 1) / nchunk0;
  const int64_t dr1 = (nr1 + nchunk1 - 1) / nchunk1;

  // The first chunk comes from our thread_id, the rest will get auto-assigned.
  int current_chunk = ith;

  while (current_chunk < nchunk0 * nchunk1) {
    const int64_t ith0 = current_chunk % nchunk0;
    const int64_t ith1 = current_chunk / nchunk0;

    const int64_t ir0_start = dr0 * ith0;
    const int64_t ir0_end = MIN(ir0_start + dr0, nr0);

    const int64_t ir1_start = dr1 * ith1;
    const int64_t ir1_end = MIN(ir1_start + dr1, nr1);

    // dot kernels can handle 1 row and col at a time, but mmla kernels can
    // process 2 rows and cols
    int64_t num_rows_per_vec_dot = vec_dot_num_rows;

    // these checks are needed to avoid crossing dim1 boundaries
    // can be optimized, but the logic would become more complicated, so keeping
    // it like this for simplicity
    if ((nr0 % 2 != 0) || (ne11 % 2 != 0) || ((ir0_end - ir0_start) % 2 != 0) ||
        ((ir1_end - ir1_start) % 2 != 0)) {
      num_rows_per_vec_dot = 1;
    }
    ggml_compute_forward_mul_mat_one_chunk(params, dst, src0->type,
                                           num_rows_per_vec_dot, ir0_start,
                                           ir0_end, ir1_start, ir1_end);

    if (nth >= nchunk0 * nchunk1) {
      break;
    }

    current_chunk = atomic_fetch_add_explicit(
        &params->threadpool->current_chunk, 1, memory_order_relaxed);
  }
}