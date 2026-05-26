/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * Scalar (portable C) implementation of q8dwconv mp8x25 (5x5 dilated layout).
 * Weight layout per packed_weights buffer with cr=8 and c_stride channel blocks:
 *   chunk0: [bias(32) | 10 weights(80)] per cr-block, total 14*c_stride bytes
 *   chunk1: [10 weights(80)]            per cr-block, total 10*c_stride bytes
 *   chunk2: [5 weights(40)]             per cr-block, total  5*c_stride bytes
 * Input pointers: input[0..9] for chunk0, input[10..19] for chunk1, input[20..24] for chunk2.
 * Bias is raw (no izp/kzp pre-adjustment), so we compute (i-izp)*(w-kzp).
 */

#include <stddef.h>
#include <stdint.h>

#include <qnnpack/q8dwconv.h>
#include <qnnpack/scalar-utils.h>


void q8dwconv_ukernel_mp8x25__scalar(
    size_t channels,
    size_t output_width,
    const uint8_t** input,
    const void* weights,
    int32_t* outacc32,
    uint8_t* output,
    size_t input_stride,
    size_t output_increment,
    const union qnnp_conv_quantization_params quantization_params[restrict static 1])
{
  const int32_t input_zero_point =
      quantization_params->scalar.input_zero_point;
  const int32_t kernel_zero_point =
      quantization_params->scalar.kernel_zero_point;
  const int32_t multiplier = quantization_params->scalar.multiplier;
  const int32_t remainder_mask = quantization_params->scalar.remainder_mask;
  const int32_t remainder_threshold =
      quantization_params->scalar.remainder_threshold;
  const uint32_t shift = quantization_params->scalar.shift;
  const int32_t smin = quantization_params->scalar.output_min_less_zero_point;
  const int32_t smax = quantization_params->scalar.output_max_less_zero_point;
  const int32_t output_zero_point = quantization_params->scalar.output_zero_point;

  /* c_stride = round up channels to multiple of cr=8 */
  const size_t c_stride = (channels + 7) & ~(size_t) 7;
  const size_t chunk1_offset = 14 * c_stride;
  const size_t chunk2_offset = 24 * c_stride;

  do {
    /* Phase 1: input[0..9], weights chunk0 with bias. */
    {
      const uint8_t* i0 = input[0];
      const uint8_t* i1 = input[1];
      const uint8_t* i2 = input[2];
      const uint8_t* i3 = input[3];
      const uint8_t* i4 = input[4];
      const uint8_t* i5 = input[5];
      const uint8_t* i6 = input[6];
      const uint8_t* i7 = input[7];
      const uint8_t* i8 = input[8];
      const uint8_t* i9 = input[9];

      const uint8_t* w = (const uint8_t*) weights;
      int32_t* outacc = outacc32;
      size_t c = channels;
      for (; c >= 8; c -= 8) {
        const int32_t* pbias = (const int32_t*) w;
        const uint8_t* pw = w + 32;
        for (size_t cc = 0; cc < 8; cc++) {
          int32_t acc = pbias[cc];
          acc += ((int32_t) i0[cc] - input_zero_point) * ((int32_t) pw[0 * 8 + cc] - kernel_zero_point);
          acc += ((int32_t) i1[cc] - input_zero_point) * ((int32_t) pw[1 * 8 + cc] - kernel_zero_point);
          acc += ((int32_t) i2[cc] - input_zero_point) * ((int32_t) pw[2 * 8 + cc] - kernel_zero_point);
          acc += ((int32_t) i3[cc] - input_zero_point) * ((int32_t) pw[3 * 8 + cc] - kernel_zero_point);
          acc += ((int32_t) i4[cc] - input_zero_point) * ((int32_t) pw[4 * 8 + cc] - kernel_zero_point);
          acc += ((int32_t) i5[cc] - input_zero_point) * ((int32_t) pw[5 * 8 + cc] - kernel_zero_point);
          acc += ((int32_t) i6[cc] - input_zero_point) * ((int32_t) pw[6 * 8 + cc] - kernel_zero_point);
          acc += ((int32_t) i7[cc] - input_zero_point) * ((int32_t) pw[7 * 8 + cc] - kernel_zero_point);
          acc += ((int32_t) i8[cc] - input_zero_point) * ((int32_t) pw[8 * 8 + cc] - kernel_zero_point);
          acc += ((int32_t) i9[cc] - input_zero_point) * ((int32_t) pw[9 * 8 + cc] - kernel_zero_point);
          outacc[cc] = acc;
        }
        i0 += 8; i1 += 8; i2 += 8; i3 += 8; i4 += 8;
        i5 += 8; i6 += 8; i7 += 8; i8 += 8; i9 += 8;
        outacc += 8;
        w += 112;
      }
      if (c != 0) {
        const int32_t* pbias = (const int32_t*) w;
        const uint8_t* pw = w + 32;
        for (size_t cc = 0; cc < c; cc++) {
          int32_t acc = pbias[cc];
          acc += ((int32_t) i0[cc] - input_zero_point) * ((int32_t) pw[0 * 8 + cc] - kernel_zero_point);
          acc += ((int32_t) i1[cc] - input_zero_point) * ((int32_t) pw[1 * 8 + cc] - kernel_zero_point);
          acc += ((int32_t) i2[cc] - input_zero_point) * ((int32_t) pw[2 * 8 + cc] - kernel_zero_point);
          acc += ((int32_t) i3[cc] - input_zero_point) * ((int32_t) pw[3 * 8 + cc] - kernel_zero_point);
          acc += ((int32_t) i4[cc] - input_zero_point) * ((int32_t) pw[4 * 8 + cc] - kernel_zero_point);
          acc += ((int32_t) i5[cc] - input_zero_point) * ((int32_t) pw[5 * 8 + cc] - kernel_zero_point);
          acc += ((int32_t) i6[cc] - input_zero_point) * ((int32_t) pw[6 * 8 + cc] - kernel_zero_point);
          acc += ((int32_t) i7[cc] - input_zero_point) * ((int32_t) pw[7 * 8 + cc] - kernel_zero_point);
          acc += ((int32_t) i8[cc] - input_zero_point) * ((int32_t) pw[8 * 8 + cc] - kernel_zero_point);
          acc += ((int32_t) i9[cc] - input_zero_point) * ((int32_t) pw[9 * 8 + cc] - kernel_zero_point);
          outacc[cc] = acc;
        }
      }
    }

    /* Phase 2: input[10..19], chunk1 (no bias). */
    {
      const uint8_t* i0 = input[10];
      const uint8_t* i1 = input[11];
      const uint8_t* i2 = input[12];
      const uint8_t* i3 = input[13];
      const uint8_t* i4 = input[14];
      const uint8_t* i5 = input[15];
      const uint8_t* i6 = input[16];
      const uint8_t* i7 = input[17];
      const uint8_t* i8 = input[18];
      const uint8_t* i9 = input[19];

      const uint8_t* w = (const uint8_t*) weights + chunk1_offset;
      int32_t* outacc = outacc32;
      size_t c = channels;
      for (; c >= 8; c -= 8) {
        const uint8_t* pw = w;
        for (size_t cc = 0; cc < 8; cc++) {
          int32_t acc = outacc[cc];
          acc += ((int32_t) i0[cc] - input_zero_point) * ((int32_t) pw[0 * 8 + cc] - kernel_zero_point);
          acc += ((int32_t) i1[cc] - input_zero_point) * ((int32_t) pw[1 * 8 + cc] - kernel_zero_point);
          acc += ((int32_t) i2[cc] - input_zero_point) * ((int32_t) pw[2 * 8 + cc] - kernel_zero_point);
          acc += ((int32_t) i3[cc] - input_zero_point) * ((int32_t) pw[3 * 8 + cc] - kernel_zero_point);
          acc += ((int32_t) i4[cc] - input_zero_point) * ((int32_t) pw[4 * 8 + cc] - kernel_zero_point);
          acc += ((int32_t) i5[cc] - input_zero_point) * ((int32_t) pw[5 * 8 + cc] - kernel_zero_point);
          acc += ((int32_t) i6[cc] - input_zero_point) * ((int32_t) pw[6 * 8 + cc] - kernel_zero_point);
          acc += ((int32_t) i7[cc] - input_zero_point) * ((int32_t) pw[7 * 8 + cc] - kernel_zero_point);
          acc += ((int32_t) i8[cc] - input_zero_point) * ((int32_t) pw[8 * 8 + cc] - kernel_zero_point);
          acc += ((int32_t) i9[cc] - input_zero_point) * ((int32_t) pw[9 * 8 + cc] - kernel_zero_point);
          outacc[cc] = acc;
        }
        i0 += 8; i1 += 8; i2 += 8; i3 += 8; i4 += 8;
        i5 += 8; i6 += 8; i7 += 8; i8 += 8; i9 += 8;
        outacc += 8;
        w += 80;
      }
      if (c != 0) {
        const uint8_t* pw = w;
        for (size_t cc = 0; cc < c; cc++) {
          int32_t acc = outacc[cc];
          acc += ((int32_t) i0[cc] - input_zero_point) * ((int32_t) pw[0 * 8 + cc] - kernel_zero_point);
          acc += ((int32_t) i1[cc] - input_zero_point) * ((int32_t) pw[1 * 8 + cc] - kernel_zero_point);
          acc += ((int32_t) i2[cc] - input_zero_point) * ((int32_t) pw[2 * 8 + cc] - kernel_zero_point);
          acc += ((int32_t) i3[cc] - input_zero_point) * ((int32_t) pw[3 * 8 + cc] - kernel_zero_point);
          acc += ((int32_t) i4[cc] - input_zero_point) * ((int32_t) pw[4 * 8 + cc] - kernel_zero_point);
          acc += ((int32_t) i5[cc] - input_zero_point) * ((int32_t) pw[5 * 8 + cc] - kernel_zero_point);
          acc += ((int32_t) i6[cc] - input_zero_point) * ((int32_t) pw[6 * 8 + cc] - kernel_zero_point);
          acc += ((int32_t) i7[cc] - input_zero_point) * ((int32_t) pw[7 * 8 + cc] - kernel_zero_point);
          acc += ((int32_t) i8[cc] - input_zero_point) * ((int32_t) pw[8 * 8 + cc] - kernel_zero_point);
          acc += ((int32_t) i9[cc] - input_zero_point) * ((int32_t) pw[9 * 8 + cc] - kernel_zero_point);
          outacc[cc] = acc;
        }
      }
    }

    /* Phase 3: input[20..24], chunk2 (no bias), then requantize and store. */
    {
      const uint8_t* i0 = input[20];
      const uint8_t* i1 = input[21];
      const uint8_t* i2 = input[22];
      const uint8_t* i3 = input[23];
      const uint8_t* i4 = input[24];

      const uint8_t* w = (const uint8_t*) weights + chunk2_offset;
      int32_t* outacc = outacc32;
      size_t c = channels;
      for (; c >= 8; c -= 8) {
        const uint8_t* pw = w;
        for (size_t cc = 0; cc < 8; cc++) {
          int32_t acc = outacc[cc];
          acc += ((int32_t) i0[cc] - input_zero_point) * ((int32_t) pw[0 * 8 + cc] - kernel_zero_point);
          acc += ((int32_t) i1[cc] - input_zero_point) * ((int32_t) pw[1 * 8 + cc] - kernel_zero_point);
          acc += ((int32_t) i2[cc] - input_zero_point) * ((int32_t) pw[2 * 8 + cc] - kernel_zero_point);
          acc += ((int32_t) i3[cc] - input_zero_point) * ((int32_t) pw[3 * 8 + cc] - kernel_zero_point);
          acc += ((int32_t) i4[cc] - input_zero_point) * ((int32_t) pw[4 * 8 + cc] - kernel_zero_point);

          const int64_t prod = (int64_t) acc * (int64_t) multiplier;
          const int32_t q31prod = (int32_t) (uint32_t) ((uint64_t) (prod + INT64_C(0x40000000)) >> 31);
          const int32_t rem = (q31prod & remainder_mask) - (int32_t) (q31prod < 0);
          int32_t out = asr_s32(q31prod, shift) + (int32_t) (rem > remainder_threshold);
          if (out < smin) out = smin;
          if (out > smax) out = smax;
          out += output_zero_point;
          output[cc] = (uint8_t) out;
        }
        i0 += 8; i1 += 8; i2 += 8; i3 += 8; i4 += 8;
        outacc += 8;
        output += 8;
        w += 40;
      }
      if (c != 0) {
        const uint8_t* pw = w;
        for (size_t cc = 0; cc < c; cc++) {
          int32_t acc = outacc[cc];
          acc += ((int32_t) i0[cc] - input_zero_point) * ((int32_t) pw[0 * 8 + cc] - kernel_zero_point);
          acc += ((int32_t) i1[cc] - input_zero_point) * ((int32_t) pw[1 * 8 + cc] - kernel_zero_point);
          acc += ((int32_t) i2[cc] - input_zero_point) * ((int32_t) pw[2 * 8 + cc] - kernel_zero_point);
          acc += ((int32_t) i3[cc] - input_zero_point) * ((int32_t) pw[3 * 8 + cc] - kernel_zero_point);
          acc += ((int32_t) i4[cc] - input_zero_point) * ((int32_t) pw[4 * 8 + cc] - kernel_zero_point);

          const int64_t prod = (int64_t) acc * (int64_t) multiplier;
          const int32_t q31prod = (int32_t) (uint32_t) ((uint64_t) (prod + INT64_C(0x40000000)) >> 31);
          const int32_t rem = (q31prod & remainder_mask) - (int32_t) (q31prod < 0);
          int32_t out = asr_s32(q31prod, shift) + (int32_t) (rem > remainder_threshold);
          if (out < smin) out = smin;
          if (out > smax) out = smax;
          out += output_zero_point;
          output[cc] = (uint8_t) out;
        }
        output += c;
      }
    }

    input = (const uint8_t**) ((uintptr_t) input + input_stride);
    output = (uint8_t*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}
