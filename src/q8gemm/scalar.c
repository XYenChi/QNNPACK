/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * Scalar (portable C) implementation of q8gemm microkernel.
 * Layout matches pack_q8gemm_w with nr=8, kr=1:
 *   For each nr-block: int32_t bias[8], then for each k: uint8_t w[8]
 */

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <qnnpack/q8gemm.h>
#include <qnnpack/scalar-utils.h>

void q8gemm_ukernel_4x8__scalar(
    size_t mr,
    size_t nr,
    size_t k,
    const uint8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    uint8_t* restrict c,
    size_t c_stride,
    const union qnnp_conv_quantization_params quantization_params[restrict static 1])
{
  assert(mr != 0);
  assert(mr <= 4);
  assert(nr != 0);
  assert(nr <= 8);
  assert(k != 0);

  const int32_t* packed_bias = (const int32_t*) w;
  int32_t vacc[4][8];
  for (size_t i = 0; i < 4; i++) {
    for (size_t j = 0; j < 8; j++) {
      vacc[i][j] = packed_bias[j];
    }
  }
  const uint8_t* packed_w = (const uint8_t*) ((uintptr_t) w + 8 * sizeof(int32_t));

  const int32_t kernel_zero_point = quantization_params->scalar.kernel_zero_point;

  const uint8_t* a0 = a;
  const uint8_t* a1 = a0 + (mr >= 2 ? a_stride : 0);
  const uint8_t* a2 = a1 + (mr > 2 ? a_stride : 0);
  const uint8_t* a3 = a2 + (mr == 4 ? a_stride : 0);

  for (size_t kk = 0; kk < k; kk++) {
    const int32_t va0 = (int32_t) a0[kk];
    const int32_t va1 = (int32_t) a1[kk];
    const int32_t va2 = (int32_t) a2[kk];
    const int32_t va3 = (int32_t) a3[kk];
    for (size_t j = 0; j < 8; j++) {
      const int32_t vb = (int32_t) packed_w[j] - kernel_zero_point;
      vacc[0][j] += va0 * vb;
      vacc[1][j] += va1 * vb;
      vacc[2][j] += va2 * vb;
      vacc[3][j] += va3 * vb;
    }
    packed_w += 8;
  }

  const int32_t multiplier = quantization_params->scalar.multiplier;
  const int32_t remainder_mask = quantization_params->scalar.remainder_mask;
  const int32_t remainder_threshold = quantization_params->scalar.remainder_threshold;
  const uint32_t shift = quantization_params->scalar.shift;
  const int32_t smin = quantization_params->scalar.output_min_less_zero_point;
  const int32_t smax = quantization_params->scalar.output_max_less_zero_point;
  const int32_t output_zero_point = quantization_params->scalar.output_zero_point;

  for (size_t i = 0; i < mr; i++) {
    for (size_t j = 0; j < nr; j++) {
      const int32_t x = vacc[i][j];
      const int64_t prod = (int64_t) x * (int64_t) multiplier;
      const int32_t q31prod = (int32_t) (uint32_t) ((uint64_t) (prod + INT64_C(0x40000000)) >> 31);
      const int32_t rem = (q31prod & remainder_mask) - (int32_t) (q31prod < 0);
      int32_t out = asr_s32(q31prod, shift) + (int32_t) (rem > remainder_threshold);
      if (out < smin)
        out = smin;
      if (out > smax)
        out = smax;
      out += output_zero_point;
      c[i * c_stride + j] = (uint8_t) out;
    }
  }
}
