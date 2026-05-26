/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * Scalar (portable C) implementations of q8gavgpool microkernels.
 * Sums m rows of n channel uint8 values, then quantizes.
 */

#include <stddef.h>
#include <stdint.h>

#include <qnnpack/q8gavgpool.h>
#include <qnnpack/scalar-utils.h>


static inline uint8_t scalar_avgpool_requantize(
    int32_t acc,
    int32_t multiplier,
    int64_t rounding,
    uint32_t right_shift,
    int32_t output_min_less_zero_point,
    int32_t output_max_less_zero_point,
    int32_t output_zero_point)
{
  const int64_t prod = (int64_t) acc * (int64_t) multiplier;
  const int64_t adjusted = prod - (int64_t) (acc < 0);
  int32_t shifted = (int32_t) asr_s64(adjusted + rounding, right_shift);
  if (shifted < output_min_less_zero_point) shifted = output_min_less_zero_point;
  if (shifted > output_max_less_zero_point) shifted = output_max_less_zero_point;
  shifted += output_zero_point;
  return (uint8_t) shifted;
}


void q8gavgpool_ukernel_up8xm__scalar(
    size_t m,
    size_t n,
    const uint8_t* x,
    size_t x_stride,
    const uint8_t* zero,
    uint8_t* y,
    const union qnnp_avgpool_quantization_params quantization_params[restrict static 1])
{
  const int32_t bias = quantization_params->scalar.bias;
  const int32_t multiplier = quantization_params->scalar.multiplier;
  const int64_t rounding = quantization_params->scalar.rounding;
  const uint32_t right_shift = quantization_params->scalar.right_shift;
  const int32_t omin = quantization_params->scalar.output_min_less_zero_point;
  const int32_t omax = quantization_params->scalar.output_max_less_zero_point;
  const int32_t ozp = quantization_params->scalar.output_zero_point;

  for (size_t c = 0; c < n; c++) {
    int32_t acc = bias;
    const uint8_t* row = x;
    for (size_t i = 0; i < m; i++) {
      acc += (int32_t) (uint32_t) row[c];
      row = (const uint8_t*) ((uintptr_t) row + x_stride);
    }
    y[c] = scalar_avgpool_requantize(
        acc, multiplier, rounding, right_shift, omin, omax, ozp);
  }
  (void) zero;
}


void q8gavgpool_ukernel_up8x7__scalar(
    size_t m,
    size_t n,
    const uint8_t* x,
    size_t x_stride,
    const uint8_t* zero,
    uint8_t* y,
    const union qnnp_avgpool_quantization_params quantization_params[restrict static 1])
{
  q8gavgpool_ukernel_up8xm__scalar(
      m, n, x, x_stride, zero, y, quantization_params);
}


void q8gavgpool_ukernel_mp8x7p7q__scalar(
    size_t m,
    size_t n,
    const uint8_t* x,
    size_t x_stride,
    const uint8_t* zero,
    int32_t* buffer,
    uint8_t* y,
    const union qnnp_avgpool_quantization_params quantization_params[restrict static 1])
{
  (void) buffer;
  q8gavgpool_ukernel_up8xm__scalar(
      m, n, x, x_stride, zero, y, quantization_params);
}
